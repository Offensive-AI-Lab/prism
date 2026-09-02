"""train.py — judge-driven GRPO training for the PRISM monitor.

Projection + LoRA monitor, matching the SFT architecture: raw hooked
activations → linear projection → norm_match → concat with retrieval_prompt token
embeds → `inputs_embeds` to the target model + LoRA.

Outer loop per step:
  1. Pull a batch of B prompts with their activations (precomputed cache, or
     extracted on the fly from the resident frozen target model).
  2. Run policy projection (grad) and ref projection (no grad) on the
     same activations; norm-match each.
  3. Build prefix embeddings = [scaled soft | retrieval_prompt token embeds] for
     both copies (one for the policy forward, one for the ref forward).
  4. Sample N candidates per prompt with the policy prefix + policy LoRA.
  5. Judge candidates → scalar reward per candidate.
  6. Build full-sequence inputs (prefix + candidate token embeds) for
     policy + ref; compute per-sequence log-probs under each.
  7. Loss = grpo / dpo / ipo; backward; step optimiser (LoRA policy
     params + projection.policy params).

Run:
    python -m prism.rl.train --sft-init-from <path>
    python -m prism.rl.train --pref-loss dpo
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Subset

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from prism.common.env import wandb_mode, load_env
from prism.common.schedule import make_lr_lambda
from prism.sft.model import (
    ActivationProjection,  # only referenced via ProjectionPair
    compute_target_norm,
    extract_activations,
    norm_match,
    register_hook,
)
from prism.activations import (
    PrecomputedActivationDataset,
    RecordDataset,
    ShardGroupedSampler,
    build_onthefly_collate_fn,
    build_precomputed_collate_fn,
    load_split_records,
)
from prism.rl import judge, losses, rollouts
from prism.rl.sampler import (
    PrioritizedSampler,
    RunningRewardTracker,
    build_weights,
    enumerate_record_ids,
)
from prism.rl.adapters import (
    POLICY_ADAPTER,
    REF_ADAPTER,
    ProjectionPair,
    adapter_scope,
    attach_policy_and_ref,
)
from prism.rl.config import RL_CONFIG
from prism.rl.data import (
    build_full_inputs,
    build_prefix_embeddings,
    select_pair_from_group,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


load_env()


# ─── Helpers ─────────────────────────────────────────────────────────────────

class _StepTimings:
    """Wall-clock breakdown of a single training step, phase by phase.

    Sync-on-exit makes CUDA-async work attributable to the right phase
    (without sync, kernel-launch time gets billed, not actual GPU time).
    Overhead is the one cuda.synchronize() per phase — measurable but small.
    """

    def __init__(self, sync_cuda: bool = True):
        self.phases: "OrderedDict[str, float]" = OrderedDict()
        self.sync_cuda = sync_cuda and torch.cuda.is_available()

    @contextmanager
    def phase(self, name: str):
        if self.sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync_cuda:
                torch.cuda.synchronize()
            self.phases[name] = self.phases.get(name, 0.0) + (time.perf_counter() - t0)

    def reset(self) -> None:
        self.phases.clear()

    def total(self) -> float:
        return sum(self.phases.values())

    def as_wandb(self, prefix: str = "timing/") -> dict[str, float]:
        return {f"{prefix}{k}_s": v for k, v in self.phases.items()}

    def as_log_str(self) -> str:
        total = self.total()
        parts = [f"{k}={v:.2f}s" for k, v in self.phases.items()]
        return f"total={total:.2f}s | " + " ".join(parts)


def _project_and_scale(
    pair: ProjectionPair,
    act: torch.Tensor,
    act_mask: torch.Tensor,
    target_norm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (pol_scaled, ref_scaled) — both already mask-zeroed."""
    pol = pair.policy(act)
    pol = norm_match(pol, target_norm)
    pol = pol * act_mask.unsqueeze(-1).to(pol.dtype)
    with torch.no_grad():
        ref = pair.ref(act)
        ref = norm_match(ref, target_norm)
        ref = ref * act_mask.unsqueeze(-1).to(ref.dtype)
    return pol, ref


def _score_candidates(
    candidates: list[list[str]],   # [B][N]
    prompts: list[str],          # [B]
    responses: list[str],        # [B]
    instruction_set_list: list[str],    # [B]
    cfg: dict,
    client,
) -> tuple[list[list[judge.CandidateScore]], list[list[list[str]]]]:
    """Run the judge on all B*N candidates.

    Returns ``(scored, gt_per_prompt)`` where
    - ``scored[b][i]`` is the full ``CandidateScore`` (reward + per-bullet
      lists + raw judge JSON) for candidate i of prompt b.
    - ``gt_per_prompt[b]`` is the GT bullet list parsed from ``instruction_set``
      (echoed back so callers don't have to re-split).
    """
    B = len(candidates)
    N = len(candidates[0]) if B else 0
    flat_prompts: list[str] = []
    flat_resps_a: list[str] = []
    flat_gt: list[list[str]] = []
    flat_cand: list[str] = []
    gt_per_prompt: list[list[str]] = []
    for b in range(B):
        gt = judge.split_instructions(instruction_set_list[b])
        gt_per_prompt.append(gt)
        for i in range(N):
            flat_prompts.append(prompts[b])
            flat_resps_a.append(responses[b])
            flat_gt.append(gt)
            flat_cand.append(candidates[b][i])

    scored_flat = judge.batch_score(
        prompts=flat_prompts,
        responses=flat_resps_a,
        gt_instructions=flat_gt,
        candidates=flat_cand,
        model=cfg["judge_model"],
        instruction_weight=cfg["instruction_weight"],
        hallucination_weight=cfg["hallucination_weight"],
        length_penalty_enabled=cfg["length_penalty_enabled"],
        length_penalty_k=cfg["length_penalty_k"],
        length_penalty_lambda=cfg["length_penalty_lambda"],
        length_penalty_under_enabled=cfg.get("length_penalty_under_enabled", False),
        length_penalty_under_k=cfg.get("length_penalty_under_k", 0.5),
        length_penalty_under_lambda=cfg.get("length_penalty_under_lambda", 0.15),
        workers=cfg["judge_workers"],
        max_retries=cfg["judge_max_retries"],
        client=client,
    )
    scored: list[list[judge.CandidateScore]] = []
    for b in range(B):
        scored.append([scored_flat[b * N + i] for i in range(N)])
    return scored, gt_per_prompt


def _rewards_from_scored(
    scored: list[list[judge.CandidateScore]],
) -> list[list[float]]:
    """Extract reward[B][N] from the score matrix."""
    return [[c.reward for c in row] for row in scored]


def _group_advantages(rewards: list[list[float]], eps: float = 1e-6) -> list[list[float]]:
    """Group-relative advantage per candidate — mirror of grpo_loss formula.

    Computed here purely for logging into judge_traces.jsonl. The loss
    recomputes its own internally so the gradient path is unaffected.
    """
    out: list[list[float]] = []
    for row in rewards:
        if not row:
            out.append([])
            continue
        m = sum(row) / len(row)
        var = sum((r - m) ** 2 for r in row) / len(row)
        s = var ** 0.5
        out.append([(r - m) / (s + eps) for r in row])
    return out


# ─── Judge trace logging ─────────────────────────────────────────────────────

class JudgeTraceWriter:
    """Append-only JSONL writer for per-(step, candidate) judge labels.

    The schema is self-contained — each line is a complete supervised
    example for downstream reward-model distillation:
    inputs (prompt, response, gt, itm_report, itm_bullets), labels
    (per-bullet score lists + raw judge details), and metadata
    (step, candidate_idx, advantage, reward_config).

    Disabled by passing ``path=None``.
    """

    def __init__(self, path: str | os.PathLike | None, reward_config: dict):
        self.path = Path(path) if path else None
        self.reward_config = dict(reward_config)
        self._fh = None
        self._n_written = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a")
            logger.info("Judge trace log → %s", self.path)

    def write_step(
        self,
        *,
        step: int,
        record_ids: list,
        prompts: list[str],
        responses: list[str],
        gt_per_prompt: list[list[str]],
        candidates: list[list[str]],
        scored: list[list[judge.CandidateScore]],
        advantages: list[list[float]],
        judge_model: str,
    ) -> None:
        if self._fh is None:
            return
        for b, row in enumerate(scored):
            for i, cs in enumerate(row):
                itm_bullets = judge.split_instructions(candidates[b][i])
                rec = {
                    "step": step,
                    "record_id": record_ids[b] if b < len(record_ids) else None,
                    "candidate_idx": i,
                    "prompt": prompts[b],
                    "response": responses[b],
                    "gt_instructions": gt_per_prompt[b],
                    "itm_bullets": itm_bullets,
                    "itm_report": candidates[b][i],
                    "instruction_scores": cs.instruction_scores,
                    "hallucination_scores": cs.hallucination_scores,
                    "instruction_details": cs.raw.get("instruction_details") if cs.raw else None,
                    "hallucination_details": cs.raw.get("hallucination_details") if cs.raw else None,
                    "mean_instruction_score": cs.mean_instruction_score,
                    "mean_hallucination_score": cs.mean_hallucination_score,
                    "length_penalty": cs.length_penalty,
                    "reward": cs.reward,
                    "advantage": advantages[b][i] if b < len(advantages) and i < len(advantages[b]) else None,
                    "judge_model": judge_model,
                    "judge_error": cs.raw.get("error") if isinstance(cs.raw, dict) else None,
                    "reward_config": self.reward_config,
                }
                self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self._n_written += 1
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            finally:
                self._fh = None
            logger.info("Judge trace log closed (%d candidate rows total)", self._n_written)


def _logp_forward(
    target_model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    grad: bool,
    length_normalize: bool = False,
    return_logits: bool = False,
    return_per_token: bool = False,
    chunk_size: int | None = None,
):
    """Returns the seq-level logp by default.

    * `return_logits=True` → (logp, logits). Used for the exact token-KL path.
    * `return_per_token=True` → (logp, token_lp, mask). Used for k3 KL.
      When this flag is on, `out.logits` is NOT returned and goes out of scope
      with the function — caller cannot hold a reference and pi/ref logits
      are freed before the loss step, which is the whole point of k3.
    Both flags can't be True at the same time.
    """
    assert not (return_logits and return_per_token), \
        "return_logits and return_per_token are mutually exclusive"
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = target_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
    if return_per_token:
        logp, token_lp, mask = losses.sequence_logprobs(
            out.logits, labels,
            length_normalize=length_normalize,
            chunk_size=chunk_size,
            return_per_token=True,
        )
        return logp, token_lp, mask
    logp = losses.sequence_logprobs(
        out.logits, labels,
        length_normalize=length_normalize,
        chunk_size=chunk_size,
    )
    if return_logits:
        return logp, out.logits
    return logp


def _save_checkpoint(target_model, projection_pair, opt_step, val_reward, cfg, path,
                     optimizer=None, scheduler=None, best_reward=None,
                     reward_tracker=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ck = {
        "lora_state": get_peft_model_state_dict(target_model, adapter_name=POLICY_ADAPTER),
        "projection_state": projection_pair.policy.state_dict(),
        "opt_step": opt_step,
        "val_reward": val_reward,
        "config": cfg,
    }
    if optimizer is not None:
        ck["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        ck["scheduler_state"] = scheduler.state_dict()
    if best_reward is not None and math.isfinite(best_reward):
        ck["best_reward"] = float(best_reward)
    # Stash the wandb run id so --resume can reattach to the same wandb run
    # instead of opening a fresh one. wandb.run is None until wandb.init()
    # has fired, which on a fresh launch happens BEFORE the first checkpoint
    # save — so this is reliably populated whenever we save mid-run.
    if wandb.run is not None:
        ck["wandb_run_id"] = wandb.run.id
    if reward_tracker is not None:
        ck["reward_tracker_state"] = reward_tracker.state_dict()
    torch.save(ck, path)
    logger.info("Checkpoint saved → %s", path)


# ─── Main ────────────────────────────────────────────────────────────────────

def resolve_data_mode(cfg: dict) -> bool:
    """Validate the data configuration; return True for ON-THE-FLY extraction.

    Exactly one of ``precomputed_dir`` (activation cache) and ``dataset_paths``
    (oracle JSONLs, activations extracted in-loop) must be set. Runs before the
    target model is loaded so a bad path fails in seconds, not minutes.
    """
    pre = cfg.get("precomputed_dir")
    paths = list(cfg.get("dataset_paths") or [])
    if pre and paths:
        raise SystemExit(
            "Both precomputed_dir and dataset_paths are set — choose one: the activation "
            "cache (--precomputed-dir / PRISM_PRECOMPUTED_DIR) or on-the-fly extraction "
            "(--dataset-paths / PRISM_DATASET_PATHS)."
        )
    if not pre and not paths:
        raise SystemExit(
            "No training data configured. Either:\n"
            "  - set PRISM_PRECOMPUTED_DIR / --precomputed-dir to an activation cache "
            "(recipes/_lib.sh do_precompute), or\n"
            "  - pass --dataset-paths <jsonl>... (or PRISM_DATASET_PATHS) for on-the-fly "
            "activation extraction."
        )
    if pre and not os.path.exists(os.path.join(pre, "manifest.json")):
        raise SystemExit(f"precomputed_dir has no manifest.json: {pre}")
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"dataset_paths not found: {missing}")
    return bool(paths)


def _extract_batch_activations(target_model, act_store: dict, batch: dict, cfg: dict, device):
    """On-the-fly activations for one batch from the resident frozen target model.

    All LoRA adapters are disabled for the forward (pure base model — the same
    computation the extractor performs), and the hook early-exits at the hook
    layer, so layers above it never run. Returns ``(act [B, N, D] bf16,
    act_mask [B, N])`` with real tokens at ``[:n]`` — the contract the
    precomputed cache satisfies.
    """
    with target_model.disable_adapter():
        act, act_mask = extract_activations(
            target_model=target_model,
            act_store=act_store,
            input_ids=batch["a_input_ids"].to(device),
            attention_mask=batch["a_attention_mask"].to(device),
            prompt_only_lens=batch["a_prompt_only_lens"],
            response_token_counts=batch["a_response_token_counts"],
            max_act_tokens=cfg["max_act_tokens"],
        )
    return act, act_mask


def train(cfg: dict):
    if not cfg.get("sft_init_from"):
        raise SystemExit(
            "sft_init_from is not set. RL initialises from an SFT checkpoint that "
            "carries lora_state + projection_state — set PRISM_SFT_INIT_FROM (or "
            "pass --sft-init-from) to the path of an SFT best.pt."
        )
    if cfg["pref_loss"] not in {"grpo", "dpo", "ipo"}:
        raise ValueError(f"Unknown pref_loss={cfg['pref_loss']!r}")
    # config.py reads PRISM_JUDGE_MODEL at import time — before load_env().
    # Re-resolve here if the import-time value was the sentinel.
    if cfg.get("judge_model", "MODEL_NOT_SET") == "MODEL_NOT_SET":
        cfg["judge_model"] = os.environ.get("PRISM_JUDGE_MODEL") or ""
    if not cfg["judge_model"]:
        raise ValueError("judge_model not set. Export PRISM_JUDGE_MODEL in .env or pass --judge-model.")
    if not cfg.get("use_projection", False):
        raise ValueError("This variant requires use_projection=True.")
    on_the_fly = resolve_data_mode(cfg)
    hook_layer = cfg["hook_layer"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s  |  pref_loss: %s", device, cfg["pref_loss"])

    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])

    # ── 1. Target model + dual LoRA ─────────────────────────────────────
    # For Qwen, load_target_model applies the transformers 5.3.0 Qwen3_5
    # compute_3d_position_ids monkeypatch: that method caches `self.rope_deltas`
    # from an earlier forward and later does `batch_size // rope_deltas.shape[0]`,
    # which yields 0 when a later forward has a SMALLER batch (dynamic sampling
    # dropping a group of N candidates → B*N shrinks to N). Forcing it to return
    # None falls through to the cache_position-derived 1D position path, which
    # handles arbitrary batch sizes. Gated on the profile — gemma-2 / Ministral
    # have no such method and are untouched.
    from prism.target_models import load_target_model, resolve_gen_eos_id

    logger.info("Loading %s …", cfg["model_id"])
    t0 = time.time()
    target_model, tokenizer, _profile = load_target_model(device)
    act_store: dict = {}
    hook_handle = None
    if on_the_fly:
        # Early-exit hook at the hook layer; inactive during rollouts/loss
        # (act_store["active"] is only True inside extract_activations).
        act_store, hook_handle = register_hook(target_model, hook_layer)
    cfg["_gen_eos_id"] = resolve_gen_eos_id(_profile, tokenizer)
    cfg["_pad_token_id"] = tokenizer.pad_token_id

    target_model.config.use_cache = False
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad_(False)

    # Gradient checkpointing — recompute activations on backward to fit larger
    # N (rollouts per prompt). ~30% slower training, ~40-50% less peak memory
    # during backward through the policy. The non-reentrant variant is what
    # PEFT + autograd want here. enable_input_require_grads() is needed
    # because the base is frozen — without it, gradients don't flow through
    # inputs_embeds to LoRA + projection.
    if cfg.get("gradient_checkpointing", True):
        target_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        target_model.enable_input_require_grads()
        logger.info("Gradient checkpointing enabled (use_reentrant=False).")
    logger.info("Target model loaded in %.1fs", time.time() - t0)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
    )
    target_model = attach_policy_and_ref(target_model, lora_cfg, cfg["sft_init_from"], device)
    target_model.print_trainable_parameters()

    # ── 2. ProjectionPair (policy + frozen ref, both SFT-init) ───────────
    projection_pair = ProjectionPair(
        dim=cfg["projection_dim"],
        sft_init_from=cfg["sft_init_from"],
        device=device,
        dtype=torch.bfloat16,
    )

    # ── 3. Target norm + data ────────────────────────────────────────────
    target_norm = compute_target_norm(target_model).to(device)
    logger.info("Target embedding norm: %.4f", target_norm.item())

    if on_the_fly:
        train_recs, val_recs, _test_recs, source_map = load_split_records(cfg, cfg.get("valid_record_ids"))
        train_ds = RecordDataset(train_recs, source_map)
        val_ds = RecordDataset(val_recs, source_map)
        train_collate = build_onthefly_collate_fn(tokenizer, cfg, source_map)
        val_collate = train_collate
        logger.info(
            "Data: ON-THE-FLY activation extraction (layer %d, last %d response tokens) "
            "from %d JSONL file(s)", hook_layer, cfg["max_act_tokens"], len(cfg["dataset_paths"]),
        )
    else:
        train_ds = PrecomputedActivationDataset(cfg["precomputed_dir"], split="train", layers=[hook_layer])
        val_ds = PrecomputedActivationDataset(cfg["precomputed_dir"], split="val", layers=[hook_layer])
        source_map = train_ds.source_map

        train_collate = build_precomputed_collate_fn(tokenizer, cfg, source_map, layers=[hook_layer], training=True)
        val_collate = build_precomputed_collate_fn(tokenizer, cfg, source_map, layers=[hook_layer], training=False)
        logger.info("Data: precomputed activation cache %s", cfg["precomputed_dir"])

    overfit_size = cfg.get("overfit_size")
    if overfit_size:
        # Sanity-check mode: train on a tiny fixed slice and use the SAME
        # slice for val. If GRPO is wired correctly the policy should
        # overfit — train+val reward climb toward the judge ceiling. If
        # they stay flat, the optimisation loop itself is broken.
        #
        # Indices are sampled RANDOMLY (seeded) rather than range(N): the
        # first N records of the precomputed dataset are often dominated by
        # one source and skew easy/short, which leaves GRPO with no
        # advantage signal (all candidates score 1.0 → reward_std=0).
        if overfit_size > len(train_ds):
            raise ValueError(f"overfit_size={overfit_size} > train set size {len(train_ds)}")
        rng = random.Random(cfg["seed"])
        slice_indices = sorted(rng.sample(range(len(train_ds)), overfit_size))
        train_ds = Subset(train_ds, slice_indices)
        val_ds = Subset(train_ds.dataset, slice_indices)
        train_sampler = None
        cfg["eval_samples"] = overfit_size
        # The for-batch loop iterates epochs * (len(train_loader)) times. With
        # only N samples cycled, we need many epochs to reach max_opt_steps.
        # Cap epochs at a high constant; max_opt_steps stops the run.
        cfg["epochs"] = max(cfg.get("epochs", 1), 10000)
        logger.info(
            "OVERFIT MODE: train+val both = %d records (random seeded indices: %s)",
            overfit_size, slice_indices,
        )
    else:
        # Shard-grouped shuffling for the cache; plain DataLoader shuffle
        # (train_sampler=None → shuffle=True below) for in-memory records.
        train_sampler = None if on_the_fly else ShardGroupedSampler(train_ds, shuffle=True, seed=cfg["seed"])

    # ── Hard-sample subset (optional) ────────────────────────────────────
    # If cfg["hard_ids_json"] points at a JSON produced by
    # prism.rl.build_hard_ids, filter train_ds to that subset.
    # Skipped under overfit_size (overfit already wraps in a Subset).
    hard_ids_path = cfg.get("hard_ids_json")
    if hard_ids_path and not overfit_size:
        payload = json.loads(Path(hard_ids_path).read_text(encoding="utf-8"))
        hard_ids = payload["record_ids"] if isinstance(payload, dict) else payload
        if not hard_ids:
            raise ValueError(f"hard_ids_json {hard_ids_path} has empty record_ids list")
        hard_ids_set = set(hard_ids)
        # Walk the dataset's per-shard meta files to map record_id → dataset idx
        # without paying the activation-load cost of __getitem__.
        rid_to_idx: dict[str, int] = {}
        if on_the_fly:
            for ds_idx, rid in enumerate(train_ds.record_ids()):
                if rid in hard_ids_set and rid not in rid_to_idx:
                    rid_to_idx[rid] = ds_idx
        else:
            last_meta_path = None
            cached_metas: list[dict] = []
            for ds_idx in range(len(train_ds)):
                _shard, meta_path, offset, _n = train_ds._index[ds_idx]
                if meta_path != last_meta_path:
                    cached_metas = json.loads(Path(meta_path).read_text(encoding="utf-8"))
                    last_meta_path = meta_path
                rid = cached_metas[offset].get("record_id", "")
                if rid in hard_ids_set and rid not in rid_to_idx:
                    rid_to_idx[rid] = ds_idx
        # Preserve the order from the hard_ids JSON (which is already stratified).
        keep_indices = [rid_to_idx[r] for r in hard_ids if r in rid_to_idx]
        missing = len(hard_ids) - len(keep_indices)
        if not keep_indices:
            raise ValueError(
                f"hard_ids_json {hard_ids_path}: 0/{len(hard_ids)} ids matched the "
                f"train split ({cfg.get('precomputed_dir') or 'on-the-fly dataset'}). Wrong split?"
            )
        if missing:
            logger.warning(
                "hard_ids_json: %d/%d ids not in train split (e.g. wrong "
                "precomputed_dir, or ids came from a val/test sweep)",
                missing, len(hard_ids),
            )
        n_before = len(train_ds)
        train_ds = Subset(train_ds, keep_indices)
        # ShardGroupedSampler operates on the underlying dataset's shard
        # boundaries; Subset would scramble those. Fall back to vanilla
        # DataLoader shuffle (the `shuffle=(train_sampler is None and ...)`
        # expression below picks this up).
        train_sampler = None
        logger.info(
            "HARD-IDS MODE: filtered train %d → %d records via %s",
            n_before, len(train_ds), hard_ids_path,
        )

    # ── Prioritized sampling (historical notes §1.8 "Fix B") ─────────────
    # Replace ShardGroupedSampler with a weighted-multinomial sampler that
    # over-samples prompts with low running-mean reward and under-samples
    # the easy/saturated ones — without fully excluding any prompt. Mutually
    # exclusive with overfit / hard_ids modes.
    prioritized_sampling = (
        bool(cfg.get("prioritized_sampling", False))
        and not overfit_size
        and not (hard_ids_path and not overfit_size)
    )
    weights_ref: list[torch.Tensor] | None = None
    reward_tracker: RunningRewardTracker | None = None
    record_ids_in_order: list[str] | None = None
    if prioritized_sampling:
        reward_tracker = RunningRewardTracker(
            alpha=float(cfg.get("prioritization_alpha", 0.3)),
        )
        # Enumerate record_ids → cached JSON next to the dataset manifest
        # so re-runs skip the (~minute) walk over all shard metas.
        rid_cache = None if on_the_fly else Path(cfg["precomputed_dir"]) / "train_record_ids.cache.json"
        record_ids_in_order = enumerate_record_ids(train_ds, cache_path=rid_cache)
        # Start with uniform weights — they kick in immediately, then the
        # tracker accumulates observations during warmup, then weight rebuild
        # at the first eval pass past warmup_steps switches us into priority mode.
        initial_weights = torch.ones(len(train_ds), dtype=torch.float32)
        weights_ref = [initial_weights]
        # Each "dataloader epoch" yields eval_every * batch_size records, so
        # the for-batch loop naturally re-enters PrioritizedSampler.__iter__
        # right after a val pass updates the weights. epochs is bumped so
        # max_opt_steps remains reachable.
        #
        # NOTE: rebuild cadence is locked to `eval_every` (not a separate
        # knob) because weights are only computed inside the eval block.
        # A separate knob would create a mismatch where weights age between
        # val passes — confusing rather than useful.
        rebuild_every = int(cfg["eval_every"])
        samples_per_iter = rebuild_every * cfg["batch_size"]
        train_sampler = PrioritizedSampler(
            num_records=len(train_ds),
            weights_ref=weights_ref,
            samples_per_iter=samples_per_iter,
            seed=cfg["seed"],
        )
        # max_opt_steps / rebuild_every gives the number of priority epochs;
        # add a generous buffer so the loop doesn't exit early.
        if cfg.get("max_opt_steps"):
            n_epochs_needed = int(cfg["max_opt_steps"]) // rebuild_every + 8
            cfg["epochs"] = max(int(cfg.get("epochs", 1)), n_epochs_needed)
        logger.info(
            "PRIORITIZED SAMPLING: alpha=%.2f, eps=%.2f, max_w=%.2f, "
            "default_mean=%.2f, warmup=%d, rebuild_every=%d, "
            "samples_per_iter=%d, dataset=%d",
            float(cfg.get("prioritization_alpha", 0.3)),
            float(cfg.get("prioritization_epsilon", 0.1)),
            float(cfg.get("prioritization_max_weight", 5.0)),
            float(cfg.get("prioritization_default_mean", 0.5)),
            int(cfg.get("prioritization_warmup_steps", 500)),
            rebuild_every, samples_per_iter, len(train_ds),
        )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], sampler=train_sampler,
        shuffle=(train_sampler is None and not overfit_size),
        collate_fn=train_collate, num_workers=2, pin_memory=True, drop_last=True,
    )
    # Val uses its own (larger) batch size — greedy 1-candidate generate with no
    # autograd and no ref forward, so memory is dominated by the target-model forward
    # over [val_batch_size, T, V]. ~6-8× speedup over train batch_size since
    # generate batches near-linearly. Defaults to 16 (config), override
    # smaller if val OOMs on long candidates.
    val_loader = DataLoader(
        val_ds, batch_size=cfg.get("val_batch_size", cfg["batch_size"]),
        shuffle=False,
        collate_fn=val_collate, num_workers=1,
    )
    logger.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    # ── 4. Optimizer ─────────────────────────────────────────────────────
    lora_params = [
        p for n, p in target_model.named_parameters()
        if p.requires_grad and f".{POLICY_ADAPTER}." in n
    ]
    if not lora_params:
        # Fallback if PEFT names adapters differently.
        lora_params = [p for p in target_model.parameters() if p.requires_grad]
    proj_params = projection_pair.trainable_parameters()

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": cfg["lr"], "name": "lora"},
            {"params": proj_params, "lr": cfg.get("projection_lr", cfg["lr"]), "name": "projection"},
        ],
        weight_decay=cfg["weight_decay"],
    )
    logger.info(
        "Trainable — LoRA: %.1fM  Projection: %.1fM",
        sum(p.numel() for p in lora_params) / 1e6,
        sum(p.numel() for p in proj_params) / 1e6,
    )

    grad_accum = cfg["grad_accum"]
    steps_per_epoch = max(1, (len(train_loader) + grad_accum - 1) // grad_accum)
    total_opt_steps = steps_per_epoch * cfg["epochs"]
    # Clip total before deriving warmup_steps so a small --max-opt-steps run
    # doesn't end up with warmup_steps >= total_opt_steps.
    max_opt_steps = cfg.get("max_opt_steps")
    if max_opt_steps:
        total_opt_steps = min(total_opt_steps, int(max_opt_steps))
    warmup_steps = int(total_opt_steps * cfg["warmup_ratio"])

    lr_lambda = make_lr_lambda(warmup_steps, total_opt_steps, cfg["min_lr_ratio"])

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ── 4.5. Resume from RL checkpoint (optional) ────────────────────────
    # If cfg["resume_from"] points at a latest.pt produced by an earlier
    # prism.rl run, restore: policy LoRA + projection_policy + optimizer +
    # scheduler + opt_step + best_reward + the wandb run id. Lets a killed
    # run continue the same wandb curve cleanly (no duplicate runs, no
    # restart of the LR schedule).
    resume_ckpt = None
    resume_path = cfg.get("resume_from")
    if resume_path:
        logger.info("Resuming from %s …", resume_path)
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        set_peft_model_state_dict(
            target_model, resume_ckpt["lora_state"], adapter_name=POLICY_ADAPTER,
        )
        projection_pair.policy.load_state_dict(resume_ckpt["projection_state"], strict=True)
        if resume_ckpt.get("optimizer_state") is not None:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        if resume_ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(resume_ckpt["scheduler_state"])
        logger.info(
            "Resumed: opt_step=%d  val_reward=%.4f  wandb_run_id=%s  best_reward=%s",
            int(resume_ckpt.get("opt_step", 0)),
            float(resume_ckpt.get("val_reward", float("nan"))),
            resume_ckpt.get("wandb_run_id"),
            resume_ckpt.get("best_reward"),
        )
        # Prioritized sampling: restore the running-reward tracker.
        # Two sources, in order of preference:
        #   1. Embedded in the resumed ckpt (modern saves include it).
        #   2. The judge_traces.jsonl alongside the ckpt (legacy/missing field).
        # Without restoration, a `--resume` launch would start over from the
        # uniform-weights warmup and undo the curriculum from the prior run.
        if reward_tracker is not None:
            tracker_state = resume_ckpt.get("reward_tracker_state")
            if tracker_state is not None:
                reward_tracker.load_state_dict(tracker_state)
                logger.info(
                    "Resumed reward tracker from ckpt: %d records",
                    len(reward_tracker.values),
                )
            else:
                traces_path = Path(resume_path).parent / "judge_traces.jsonl"
                if traces_path.exists():
                    reward_tracker = RunningRewardTracker.from_judge_traces(
                        traces_path,
                        alpha=float(cfg.get("prioritization_alpha", 0.3)),
                    )

    # ── 5. WandB + judge client ──────────────────────────────────────────
    if wandb.run is None:
        init_kwargs = dict(
            project=cfg["wandb_project"],
            entity=cfg.get("wandb_entity"),
            config=cfg,
            name=cfg.get("wandb_run_name"),
        )
        if resume_ckpt is not None and resume_ckpt.get("wandb_run_id"):
            # resume="allow" — re-attach if the id exists, otherwise start
            # fresh under that id. "must" would error if it didn't exist;
            # not what we want for a `--resume <stale-ckpt>` scenario.
            init_kwargs["id"] = resume_ckpt["wandb_run_id"]
            init_kwargs["resume"] = "allow"
        if wandb_mode():
            init_kwargs["mode"] = wandb_mode()
        wandb.init(**init_kwargs)
    else:
        wandb.config.update(cfg, allow_val_change=True)
    judge_client = judge._make_client()

    # ── 6. Loop ──────────────────────────────────────────────────────────
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    best_path = os.path.join(cfg["checkpoint_dir"], "best.pt")
    latest_path = os.path.join(cfg["checkpoint_dir"], "latest.pt")
    best_reward = -float("inf")
    opt_step = 0
    micro_step = 0
    # Last observed val/reward_mean — stamped into mid-interval latest.pt
    # saves so the checkpoint still carries a meaningful val_reward field
    # (the most recent eval, not the one at exactly this step).
    last_val_reward = float("nan")
    if resume_ckpt is not None:
        opt_step = int(resume_ckpt.get("opt_step", 0))
        if resume_ckpt.get("best_reward") is not None:
            best_reward = float(resume_ckpt["best_reward"])
        if resume_ckpt.get("val_reward") is not None:
            last_val_reward = float(resume_ckpt["val_reward"])
        # Free the ckpt dict — the tensors are already on-device in model/opt.
        resume_ckpt = None
        # Roll the PrioritizedSampler's epoch counter forward so post-resume
        # sampling doesn't replay the same indices the original run drew on
        # its very first epoch. With samples_per_iter aligned to eval_every,
        # opt_step // eval_every is the natural epoch index.
        if prioritized_sampling and isinstance(train_sampler, PrioritizedSampler):
            train_sampler.set_epoch(opt_step // int(cfg["eval_every"]))
            logger.info(
                "PrioritizedSampler epoch fast-forwarded to %d on resume",
                train_sampler.epoch,
            )

    # Judge-trace writer: per-(step, candidate) JSONL log of full judge
    # outputs (inputs + per-bullet labels + reward + advantage). Co-located
    # with the checkpoint directory so the run's artifacts travel together.
    # Used later as supervised data for distilling the LLM judge into a
    # small reward model. Disabled with --no-judge-log.
    judge_log_path = None
    if cfg.get("judge_log_enabled", True):
        judge_log_path = Path(cfg["checkpoint_dir"]) / "judge_traces.jsonl"
    reward_config_snapshot = {
        k: cfg.get(k) for k in (
            "instruction_weight", "hallucination_weight",
            "length_penalty_enabled", "length_penalty_k", "length_penalty_lambda",
            "length_penalty_under_enabled", "length_penalty_under_k",
            "length_penalty_under_lambda",
        )
    }
    judge_trace_writer = JudgeTraceWriter(judge_log_path, reward_config_snapshot)
    # max_opt_steps already applied to total_opt_steps above (before warmup_steps).

    # Keep the target model + projection in eval mode throughout RL: this disables
    # LoRA dropout (and projection dropout if any), which would otherwise
    # randomise the policy/ref logp difference and inflate the KL term.
    # Optimiser.step() still updates parameters — train/eval flag only
    # affects dropout-like layers, not gradient flow.
    target_model.eval()
    target_model.set_adapter(POLICY_ADAPTER)
    projection_pair.policy.eval()

    # ── Async pipeline ──────────────────────────────────────────────────────
    # We submit the judge call for batch K to a background thread, then
    # immediately start gen for batch K+1 on the GPU. When gen K+1 finishes,
    # we await judge K (usually already done) and run fwd/bwd for K. Net per
    # opt_step in steady state: max(gen, judge) + fwd_bwd  instead of
    # gen + judge + fwd_bwd. Single-worker executor keeps semantics simple
    # (one step in flight; judge.batch_score still uses judge_workers=12
    # internally for the 8 candidates within a step).
    judge_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="judge_pipe")
    pending: dict | None = None

    def _complete_deferred(state: dict) -> bool:
        """Consume one pending state: trace_write + fwd + bwd + opt + log + eval.

        Returns True iff max_opt_steps was reached (caller breaks).
        """
        nonlocal opt_step, micro_step, best_reward, last_val_reward

        timings = state["timings"]
        batch = state["batch"]
        act = state["act"]
        act_mask = state["act_mask"]
        chat_pref_ids = state["chat_pref_ids"]
        chat_pref_mask = state["chat_pref_mask"]
        pol_scaled = state["pol_scaled"]
        ref_scaled = state["ref_scaled"]
        candidates = state["candidates"]
        candidate_token_ids = state["candidate_token_ids"]

        # 6.4 Judge — await result from background thread.
        with timings.phase("judge_wait"):
            scored, gt_per_prompt = state["judge_future"].result()
        rewards = _rewards_from_scored(scored)
        n_judge_errors = sum(
            1 for row in scored for cs in row
            if isinstance(cs.raw, dict) and cs.raw.get("error")
        )
        if n_judge_errors:
            logger.warning(
                "[step %d] %d/%d candidates returned judge errors — rewards forced to 0",
                opt_step, n_judge_errors, sum(len(r) for r in scored),
            )

        # 6.4a-pre Update the running-reward tracker (historical notes §1.8).
        # Reads the raw per-prompt rewards from THIS batch and folds them into the EMA
        # used by PrioritizedSampler to weight future draws. Skipped when the
        # judge errored — those rewards are forced-zero, not real signal.
        if reward_tracker is not None and not n_judge_errors:
            batch_rids = batch.get("record_ids") or []
            reward_tracker.update_batch(
                list(batch_rids), rewards, skip_if_all_zero=True,
            )

        # 6.4a Trace write
        with timings.phase("trace_write"):
            advantages = _group_advantages(rewards)
            batch_record_ids = batch.get("record_ids")
            if batch_record_ids is None:
                batch_record_ids = [None] * len(batch["prompts"])
            judge_trace_writer.write_step(
                step=opt_step,
                record_ids=batch_record_ids,
                prompts=batch["prompts"],
                responses=batch["responses"],
                gt_per_prompt=gt_per_prompt,
                candidates=candidates,
                scored=scored,
                advantages=advantages,
                judge_model=cfg["judge_model"],
            )

        # 6.4b Dynamic sampling — per-group filter applied AFTER judging,
        # BEFORE the expensive policy/ref forwards. Drops two failure modes
        # of GRPO near the reward extremes (DAPO-style, historical notes §1.7):
        #   1. group_std < ε  → all-tied. Advantages collapse to ~0; the KL
        #      term still applies → "free" pull toward ref. Net-negative.
        #   2. group_mean > τ → near judge ceiling. One bullet stochastically
        #      flipping 1.0→0.5 gives a huge spurious negative advantage,
        #      pushing the policy AWAY from near-perfect responses. Also
        #      net-negative.
        # Both knobs default-off (min_std=0, max_mean≥1.0+) via config.
        N = cfg["n_candidates"]
        B = act.shape[0]
        is_grpo = cfg["pref_loss"] == "grpo"
        ds_min_std = float(cfg.get("dynamic_sampling_min_std", 0.0))
        ds_max_mean = float(cfg.get("dynamic_sampling_max_mean", 2.0))
        ds_enabled = is_grpo and (ds_min_std > 0.0 or ds_max_mean < 1.0)
        groups_kept: list[int] = list(range(B))
        ds_drop_low_std = 0
        ds_drop_high_mean = 0
        if ds_enabled:
            kept: list[int] = []
            for b, row in enumerate(rewards):
                if len(row) < 2:
                    kept.append(b)
                    continue
                arr = torch.tensor(row, dtype=torch.float32)
                g_mean = float(arr.mean())
                g_std = float(arr.std(unbiased=False))
                if g_std < ds_min_std:
                    ds_drop_low_std += 1
                    continue
                if g_mean > ds_max_mean:
                    ds_drop_high_mean += 1
                    continue
                kept.append(b)
            groups_kept = kept
            if ds_drop_low_std or ds_drop_high_mean:
                logger.info(
                    "[step %d] dynamic-sampling: kept %d/%d groups "
                    "(dropped low-std=%d, high-mean=%d)",
                    opt_step, len(kept), B, ds_drop_low_std, ds_drop_high_mean,
                )
        if not groups_kept:
            logger.info(
                "[step %d] dynamic-sampling dropped ALL %d groups — skipping opt step",
                opt_step, B,
            )
            wandb.log({
                "train/ds_drop_low_std": ds_drop_low_std,
                "train/ds_drop_high_mean": ds_drop_high_mean,
                "train/ds_kept_groups": 0,
                "train/ds_total_groups": B,
                "train/ds_step_skipped": 1,
                "opt_step": opt_step,
            }, step=opt_step)
            return False

        # 6.5 Build full-sequence inputs + compute logps. Use ACTUAL sampled
        # token ids (detokenize→retokenize is not round-trip stable, would
        # score a different sequence than the one that earned the reward).
        flat_cand_token_ids: list[list[int]] = []
        flat_act_idx: list[int] = []
        group_ids: list[int] = []
        for b in groups_kept:
            for i in range(N):
                flat_cand_token_ids.append(candidate_token_ids[b][i])
                flat_act_idx.append(b)
                group_ids.append(b)
        act_idx_t = torch.tensor(flat_act_idx, dtype=torch.long, device=device)
        group_ids_t = torch.tensor(group_ids, dtype=torch.long, device=device)
        rewards_flat = torch.tensor(
            [r for b in groups_kept for r in rewards[b]],
            dtype=torch.float32, device=device,
        )

        pol_scaled_rep = pol_scaled[act_idx_t]
        ref_scaled_rep = ref_scaled[act_idx_t]
        act_mask_rep = act_mask[act_idx_t]
        chat_pref_ids_rep = chat_pref_ids[act_idx_t]
        chat_pref_mask_rep = chat_pref_mask[act_idx_t]

        # For GRPO, length-normalize logp so the policy-gradient signal isn't
        # dominated by candidate length (sum-logp scales with response tokens,
        # swamping the reward-driven advantage). For DPO/IPO keep sum-logp —
        # those losses are pairwise differences and the paper convention is
        # sum-over-response. See the historical notes §1.4.
        length_normalize = is_grpo and cfg.get("length_normalize_logp", True)

        # Choose KL estimator (historical notes §1.4 / §1.6):
        #   "token_exact" — exact per-token KL over full vocab. Most accurate
        #                   but memory-heavy (holds pi_logits + ref_logits).
        #   "k3"          — Schulman's k3 = (r-1) - log(r) at sampled tokens.
        #                   Production-standard, ~3-5 GB lighter at loss step.
        #                   Lets us scale N past 4 on 95 GB GPU.
        #   "seq_proxy"   — legacy `(pi_logp - ref_logp).mean()`. Kept for
        #                   ablation; not recommended (can go negative).
        kl_estimator = cfg.get("kl_estimator", "token_exact") if is_grpo else None
        use_token_kl = (kl_estimator == "token_exact")
        use_k3 = (kl_estimator == "k3")
        # Chunk sizes for memory-bounded loss computation. fp32 log_softmax
        # over the full vocab (Qwen3.5: 152K) dominates loss-step memory; without
        # chunking it OOMs at N≥6 on a 95 GB GPU. log_softmax is row-
        # independent in the batch dim and the KL sum is over response
        # tokens — both safe to chunk.
        logp_chunk = cfg.get("logp_chunk_size") if is_grpo else None
        kl_chunk = cfg.get("kl_chunk_size") if use_token_kl else None

        # Policy forward (grad ON)
        with timings.phase("fwd_policy"):
            pol_inputs = build_full_inputs(
                pol_scaled_rep, act_mask_rep, chat_pref_ids_rep, chat_pref_mask_rep,
                flat_cand_token_ids, target_model, pad_id=tokenizer.pad_token_id,
            )
            target_model.set_adapter(POLICY_ADAPTER)
            pol_out = _logp_forward(
                target_model,
                pol_inputs["inputs_embeds"], pol_inputs["attention_mask"], pol_inputs["labels"],
                grad=True,
                length_normalize=length_normalize,
                return_logits=use_token_kl,
                return_per_token=use_k3,
                chunk_size=logp_chunk,
            )
            if use_token_kl:
                pi_logp, pi_logits = pol_out
                pi_token_lp, token_mask = None, None
            elif use_k3:
                pi_logp, pi_token_lp, token_mask = pol_out
                pi_logits = None
            else:
                pi_logp = pol_out
                pi_logits, pi_token_lp, token_mask = None, None, None

        # Ref forward (grad OFF, ref projection + ref LoRA)
        with timings.phase("fwd_ref"):
            with torch.no_grad():
                ref_inputs = build_full_inputs(
                    ref_scaled_rep, act_mask_rep, chat_pref_ids_rep, chat_pref_mask_rep,
                    flat_cand_token_ids, target_model, pad_id=tokenizer.pad_token_id,
                )
            with adapter_scope(target_model, REF_ADAPTER):
                ref_out = _logp_forward(
                    target_model,
                    ref_inputs["inputs_embeds"], ref_inputs["attention_mask"], ref_inputs["labels"],
                    grad=False,
                    length_normalize=length_normalize,
                    return_logits=use_token_kl,
                    return_per_token=use_k3,
                    chunk_size=logp_chunk,
                )
            if use_token_kl:
                ref_logp, ref_logits = ref_out
                ref_token_lp = None
            elif use_k3:
                ref_logp, ref_token_lp, _ = ref_out
                ref_logits = None
            else:
                ref_logp = ref_out
                ref_logits, ref_token_lp = None, None

        # 6.6 Loss
        with timings.phase("loss_backward"):
            if is_grpo:
                loss, metrics = losses.grpo_loss(
                    pi_logp, ref_logp, rewards_flat, group_ids_t,
                    kl_coef=cfg["kl_coef"],
                    policy_logits=pi_logits,
                    ref_logits=ref_logits,
                    labels=pol_inputs["labels"] if use_token_kl else None,
                    kl_chunk_size=kl_chunk,
                    pi_token_logp=pi_token_lp,
                    ref_token_logp=ref_token_lp,
                    token_mask=token_mask,
                )
            else:
                chosen_idx: list[int] = []
                rejected_idx: list[int] = []
                for b in range(B):
                    pair = select_pair_from_group(rewards[b], min_margin=cfg["min_margin"])
                    if pair is None:
                        continue
                    bi, wi = pair
                    chosen_idx.append(b * N + bi)
                    rejected_idx.append(b * N + wi)
                if not chosen_idx:
                    logger.info("[step %d] no valid pairs (ties or below margin) — skipping", opt_step)
                    return False
                ci = torch.tensor(chosen_idx, device=device, dtype=torch.long)
                ri = torch.tensor(rejected_idx, device=device, dtype=torch.long)
                if cfg["pref_loss"] == "dpo":
                    loss, metrics = losses.dpo_loss(
                        pi_logp[ci], pi_logp[ri], ref_logp[ci], ref_logp[ri], beta=cfg["beta"],
                    )
                else:
                    loss, metrics = losses.ipo_loss(
                        pi_logp[ci], pi_logp[ri], ref_logp[ci], ref_logp[ri], beta=cfg["beta"],
                    )

            (loss / grad_accum).backward()

        micro_step += 1

        if micro_step % grad_accum == 0:
            with timings.phase("opt"):
                gnorm_lora = torch.nn.utils.clip_grad_norm_(lora_params, cfg["grad_clip"]).item()
                gnorm_proj = torch.nn.utils.clip_grad_norm_(proj_params, cfg["grad_clip"]).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                # Defragment between steps. Variable-length candidate batches
                # leave the allocator with growing holes that eventually OOM
                # the next long-sequence batch even though raw free memory is
                # adequate. empty_cache() is cheap (microseconds) and avoids
                # the "94 GB in use, 234 MB unallocated, can't fit 1 GB"
                # fragmentation pattern.
                if cfg.get("empty_cache_each_step", True):
                    torch.cuda.empty_cache()
            opt_step += 1

            if opt_step % cfg["log_every"] == 0:
                reward_mean = float(rewards_flat.mean().item())
                reward_std = float(rewards_flat.std(unbiased=False).item()) if rewards_flat.numel() > 1 else 0.0
                payload = {
                    "train/loss": float(loss.detach().item()),
                    "train/grad_norm_lora": gnorm_lora,
                    "train/grad_norm_proj": gnorm_proj,
                    "train/lr_lora": optimizer.param_groups[0]["lr"],
                    "train/lr_proj": optimizer.param_groups[1]["lr"],
                    "train/reward_mean": reward_mean,
                    "train/reward_std": reward_std,
                    "train/ds_drop_low_std": ds_drop_low_std,
                    "train/ds_drop_high_mean": ds_drop_high_mean,
                    "train/ds_kept_groups": len(groups_kept),
                    "train/ds_total_groups": B,
                    "opt_step": opt_step,
                    **{f"train/{k}": v for k, v in metrics.items()},
                    **timings.as_wandb(),
                    "timing/step_total_s": timings.total(),
                }
                wandb.log(payload, step=opt_step)
                logger.info(
                    "[step %d/%d] loss=%.4f reward=%.3f±%.3f gn_lora=%.2f gn_proj=%.2f",
                    opt_step, total_opt_steps, payload["train/loss"],
                    reward_mean, reward_std, gnorm_lora, gnorm_proj,
                )
                logger.info("  timings: %s", timings.as_log_str())

            # Periodic latest.pt save — decoupled from eval cadence so a
            # crash loses at most `save_every` steps of work, not
            # `eval_every`. Skip when this step will also trigger an eval
            # (which does its own save with the fresh val_reward).
            save_every = int(cfg.get("save_every", cfg["eval_every"]))
            if (
                save_every > 0
                and opt_step % save_every == 0
                and opt_step % cfg["eval_every"] != 0
            ):
                _save_checkpoint(
                    target_model, projection_pair, opt_step, last_val_reward,
                    cfg, latest_path, optimizer, scheduler, best_reward=best_reward,
                    reward_tracker=reward_tracker,
                )

            if opt_step % cfg["eval_every"] == 0:
                val_out = _eval_judge_reward(
                    target_model, tokenizer, projection_pair, target_norm,
                    val_loader, hook_layer, cfg, device, judge_client,
                    source_map=source_map, act_store=act_store,
                )
                val_reward = val_out["reward_mean"]
                # Track latest known val for the next mid-interval save.
                last_val_reward = val_reward
                # Rebuild prioritized-sampling weights from the current tracker
                # state (historical notes §1.8). Uniform during warmup → priority-weighted
                # after warmup_steps. The new tensor is swapped into weights_ref
                # so the next PrioritizedSampler.__iter__ picks it up. Logged
                # weight-distribution percentiles so wandb shows the curriculum
                # tightening over training.
                if reward_tracker is not None and weights_ref is not None and record_ids_in_order is not None:
                    warmup = int(cfg.get("prioritization_warmup_steps", 500))
                    if opt_step >= warmup:
                        new_weights = build_weights(
                            record_ids_in_order, reward_tracker,
                            epsilon=float(cfg["prioritization_epsilon"]),
                            max_weight=float(cfg["prioritization_max_weight"]),
                            default_mean=float(cfg["prioritization_default_mean"]),
                        )
                    else:
                        new_weights = torch.ones(len(record_ids_in_order), dtype=torch.float32)
                    weights_ref[0] = new_weights
                    q = torch.quantile(new_weights, torch.tensor([0.5, 0.9, 0.99]))
                    logger.info(
                        "Prioritized weights rebuilt: tracker=%d records, "
                        "w_p50=%.2f w_p90=%.2f w_p99=%.2f max=%.2f",
                        len(reward_tracker.values),
                        float(q[0]), float(q[1]), float(q[2]),
                        float(new_weights.max()),
                    )
                    wandb.log({
                        "prio/n_tracked": len(reward_tracker.values),
                        "prio/weight_p50": float(q[0]),
                        "prio/weight_p90": float(q[1]),
                        "prio/weight_p99": float(q[2]),
                        "prio/weight_max": float(new_weights.max()),
                    }, step=opt_step)
                wandb.log({
                    "val/reward_mean": val_reward,
                    "val/n_scored": val_out["n_scored"],
                    "val/samples": val_out["samples_table"],
                    "opt_step": opt_step,
                }, step=opt_step)
                logger.info("[VAL] reward_mean=%.4f  n=%d", val_reward, val_out["n_scored"])
                _save_checkpoint(target_model, projection_pair, opt_step, val_reward, cfg, latest_path,
                                 optimizer, scheduler, best_reward=best_reward,
                                 reward_tracker=reward_tracker)
                if val_reward > best_reward:
                    best_reward = val_reward
                    # Each new best gets its own filename so we accumulate
                    # the full trajectory instead of overwriting the prior
                    # best. The unsuffixed `best.pt` is updated to a symlink
                    # pointing at the latest best — gives a stable handle
                    # for the demo app / downstream consumers.
                    versioned_best = os.path.join(
                        cfg["checkpoint_dir"],
                        f"best_step{opt_step}_v{val_reward:.4f}.pt",
                    )
                    _save_checkpoint(target_model, projection_pair, opt_step, val_reward, cfg, versioned_best,
                                     optimizer, scheduler, best_reward=best_reward,
                                     reward_tracker=reward_tracker)
                    try:
                        if os.path.lexists(best_path):
                            os.unlink(best_path)
                        os.symlink(os.path.basename(versioned_best), best_path)
                    except OSError as e:
                        # Filesystem doesn't support symlinks (rare on NFS,
                        # never seen on this volume) → log and continue;
                        # the versioned file is still there to point at.
                        logger.warning("Could not update best.pt symlink: %s", e)
                    logger.info(
                        "  ★ new best val reward: %.4f → %s",
                        best_reward, os.path.basename(versioned_best),
                    )

            if max_opt_steps and opt_step >= max_opt_steps:
                logger.info("Reached max_opt_steps=%d — stopping.", max_opt_steps)
                return True
        return False

    for epoch in range(cfg["epochs"]):
        logger.info("═══ Epoch %d/%d ═══", epoch + 1, cfg["epochs"])
        for batch in train_loader:
            timings = _StepTimings()

            # ── Stage A (GPU): prepare CURRENT batch ────────────────────────
            with timings.phase("data_to_gpu"):
                if not on_the_fly:
                    act = batch["precomputed_acts"][hook_layer].to(device)
                    act_mask = batch["act_masks"].to(device)
                chat_pref_ids = batch["chat_prefix_input_ids"].to(device)
                chat_pref_mask = batch["chat_prefix_attention_mask"].to(device)
            if on_the_fly:
                with timings.phase("extract"):
                    act, act_mask = _extract_batch_activations(target_model, act_store, batch, cfg, device)

            with timings.phase("project"):
                pol_scaled, ref_scaled = _project_and_scale(projection_pair, act, act_mask, target_norm)

            with timings.phase("prefix_build"):
                with torch.no_grad():
                    pol_prefix_embeds, pol_prefix_mask, _ = build_prefix_embeddings(
                        pol_scaled.detach(), act_mask, chat_pref_ids, chat_pref_mask, target_model,
                    )

            with timings.phase("gen"):
                target_model.set_adapter(POLICY_ADAPTER)
                candidates, candidate_token_ids = rollouts.generate_candidates(
                    target_model, tokenizer,
                    prefix_embeds=pol_prefix_embeds,
                    prefix_attention_mask=pol_prefix_mask,
                    n_candidates=cfg["n_candidates"],
                    max_new_tokens=cfg["gen_max_new_tokens"],
                    temperature=cfg["gen_temperature"],
                    top_p=cfg["gen_top_p"],
                    eos_token_id=cfg.get("_gen_eos_id"),
                    pad_token_id=cfg.get("_pad_token_id"),
                )

            # ── Stage B: submit CURRENT batch's judge in background ─────────
            cur_state = {
                "batch": batch,
                "act": act, "act_mask": act_mask,
                "chat_pref_ids": chat_pref_ids, "chat_pref_mask": chat_pref_mask,
                "pol_scaled": pol_scaled, "ref_scaled": ref_scaled,
                "candidates": candidates, "candidate_token_ids": candidate_token_ids,
                "timings": timings,
            }
            cur_state["judge_future"] = judge_executor.submit(
                _score_candidates,
                candidates, batch["prompts"], batch["responses"],
                batch["instruction_sets"], cfg, judge_client,
            )

            # ── Stage C: finish PREVIOUS batch's deferred fwd/bwd ───────────
            if pending is not None:
                should_stop = _complete_deferred(pending)
                if should_stop:
                    pending = cur_state  # cur judge is still in flight — drain in finally
                    break

            pending = cur_state

        if max_opt_steps and opt_step >= max_opt_steps:
            break

    # Drain the final pending batch so its update lands (unless we already
    # hit max_opt_steps inside _complete_deferred — that case skips the
    # extra step intentionally).
    if pending is not None and not (max_opt_steps and opt_step >= max_opt_steps):
        _complete_deferred(pending)

    judge_executor.shutdown(wait=True)
    judge_trace_writer.close()
    logger.info("=" * 60)
    logger.info("Training complete. Best val reward: %.4f", best_reward)
    wandb.finish()


# ─── Validation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_judge_reward(
    target_model, tokenizer, projection_pair, target_norm,
    val_loader, hook_layer, cfg, device, judge_client, source_map=None,
    act_store=None,
) -> dict:
    """Greedy-rollout each val record, judge it, return mean reward + sample table.

    Returns dict with:
      "reward_mean": float
      "samples_table": wandb.Table — first `log_sample_count` val rollouts
                                     for spot-checking generation quality
      "n_scored": int
    """
    target_model.set_adapter(POLICY_ADAPTER)
    inv_source_map = {v: k for k, v in (source_map or {}).items()}

    total_reward = 0.0
    n_scored = 0
    n_seen = 0

    log_sample_count = cfg.get("log_sample_count", 5)
    table_rows: list[tuple] = []

    for batch in val_loader:
        if n_seen >= cfg["eval_samples"]:
            break
        if "precomputed_acts" in batch:
            act = batch["precomputed_acts"][hook_layer].to(device)
            act_mask = batch["act_masks"].to(device)
        else:
            act, act_mask = _extract_batch_activations(target_model, act_store, batch, cfg, device)
        chat_pref_ids = batch["chat_prefix_input_ids"].to(device)
        chat_pref_mask = batch["chat_prefix_attention_mask"].to(device)

        pol_out = projection_pair.policy(act)
        pol_scaled = norm_match(pol_out, target_norm) * act_mask.unsqueeze(-1).to(pol_out.dtype)

        prefix_embeds, prefix_mask, _ = build_prefix_embeddings(
            pol_scaled, act_mask, chat_pref_ids, chat_pref_mask, target_model,
        )
        cands, _cand_ids = rollouts.generate_candidates(
            target_model, tokenizer,
            prefix_embeds=prefix_embeds, prefix_attention_mask=prefix_mask,
            n_candidates=1,
            max_new_tokens=cfg["gen_max_new_tokens"],
            temperature=0.0, top_p=1.0, do_sample=False,
            eos_token_id=cfg.get("_gen_eos_id"),
            pad_token_id=cfg.get("_pad_token_id"),
        )
        scored, _gt = _score_candidates(
            cands, batch["prompts"], batch["responses"],
            batch["instruction_sets"], cfg, judge_client,
        )
        rewards = _rewards_from_scored(scored)
        for row in rewards:
            total_reward += sum(row)
            n_scored += len(row)

        # Collect first log_sample_count rollouts for the spot-check table.
        B = act.shape[0]
        sids = batch.get("source_ids")
        sids_list = sids.tolist() if sids is not None else [-1] * B
        for b_idx in range(B):
            if len(table_rows) >= log_sample_count:
                break
            cs = scored[b_idx][0]   # greedy → exactly one candidate
            source = inv_source_map.get(sids_list[b_idx], "unknown")
            rids = batch.get("record_ids")
            record_id = rids[b_idx] if rids is not None else ""
            table_rows.append((
                source,
                record_id,
                batch["prompts"][b_idx],
                batch["responses"][b_idx],
                batch["instruction_sets"][b_idx],
                cands[b_idx][0],
                round(cs.reward, 4),
                round(cs.mean_instruction_score, 4),
                round(cs.mean_hallucination_score, 4),
                round(cs.length_penalty, 4),
                ",".join(f"{s:.1f}" for s in cs.instruction_scores),
                ",".join(f"{s:.1f}" for s in cs.hallucination_scores),
            ))

        n_seen += B

    columns = [
        "source", "record_id", "prompt", "response", "ground_truth",
        "generated", "reward", "mean_instruction_score", "mean_hallucination_score",
        "length_penalty", "instruction_scores", "hallucination_scores",
    ]
    samples_table = wandb.Table(columns=columns)
    for row in table_rows:
        samples_table.add_data(*row)

    return {
        "reward_mean": total_reward / max(1, n_scored),
        "samples_table": samples_table,
        "n_scored": n_scored,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="prism.rl: judge-driven GRPO on the projection+LoRA monitor")
    p.add_argument("--sft-init-from", type=str, default=None)
    p.add_argument("--pref-loss", type=str, default=None, choices=["grpo", "dpo", "ipo"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--n-candidates", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--judge-model", type=str, default=None)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--precomputed-dir", type=str, default=None,
                   help="Override cfg['precomputed_dir']. Must contain a manifest.json "
                        "produced by prism.activations.extract with matching "
                        "hook_layer/num_tokens/dtype as the SFT checkpoint.")
    p.add_argument("--dataset-paths", nargs="+", default=None,
                   help="Oracle JSONL files for ON-THE-FLY activation extraction from the "
                        "resident target model (alternative to --precomputed-dir; "
                        "env: PRISM_DATASET_PATHS, os.pathsep-separated).")
    p.add_argument("--valid-record-ids", type=str, default=None,
                   help="valid_record_ids.json mask for --dataset-paths (default: auto-detect "
                        "next to the JSONL files; 'none' to disable).")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--max-opt-steps", type=int, default=None,
                   help="Hard cap on optimiser steps (e.g. 5000). Stops training early when reached.")
    p.add_argument("--no-judge-log", action="store_true",
                   help="Disable per-(step, candidate) JSONL trace of judge outputs.")
    p.add_argument("--overfit-size", type=int, default=None,
                   help="Sanity-check mode: train + val on N seeded-random samples cycled. "
                        "Reward should climb toward the judge ceiling. Forces eval_samples=N.")
    p.add_argument("--eval-every", type=int, default=None,
                   help="Override eval cadence (config default = 200). "
                        "For overfit probes, set to 25 to track val trajectory.")
    p.add_argument("--kl-estimator", type=str, default=None,
                   choices=["token_exact", "k3", "seq_proxy"],
                   help="KL estimator for GRPO (cfg default = token_exact). "
                        "Use 'k3' for the memory-light Schulman estimator that "
                        "lets N scale past 4 on a 95 GB GPU.")
    p.add_argument("--gen-max-new-tokens", type=int, default=None,
                   help="Override rollout max-new-tokens (cfg default = 192). "
                        "Cap to 128–144 if N≥6 is OOMing on long-candidate "
                        "outlier batches.")
    p.add_argument("--kl-coef", type=float, default=None,
                   help="Override cfg['kl_coef']. Lower (e.g. 0.05) to let the "
                        "policy drift further from the SFT ref — useful when "
                        "val plateaus early at a near-ceiling reward, which "
                        "typically signals a KL-pinned regime.")
    p.add_argument("--ds-min-std", type=float, default=None,
                   help="Dynamic sampling: drop groups whose reward std is "
                        "below this. 0 disables. Typical: 0.05.")
    p.add_argument("--ds-max-mean", type=float, default=None,
                   help="Dynamic sampling: drop groups whose reward mean is "
                        "above this (near-ceiling, judge-noise-dominated). "
                        "Set ≥1.0 to disable. Typical: 0.95.")
    p.add_argument("--grad-accum", type=int, default=None,
                   help="Override cfg['grad_accum']. Averages gradients across "
                        "this many micro-batches per optimiser step. NB: "
                        "max_opt_steps caps OPTIMIZER steps, so bumping "
                        "grad_accum also requires bumping --epochs to keep "
                        "the same total opt steps (steps_per_epoch shrinks).")
    p.add_argument("--hard-ids-json", type=str, default=None,
                   help="Path to a JSON produced by "
                        "prism.rl.build_hard_ids. Filters the "
                        "train dataset to only those record_ids before "
                        "DataLoader construction. Mutually exclusive with "
                        "--overfit-size (overfit takes precedence).")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a prism.rl latest.pt to resume from. "
                        "Restores LoRA(policy) + projection + optimizer + "
                        "scheduler + opt_step + best_reward, and re-attaches "
                        "to the same wandb run id stored in the checkpoint "
                        "(so the curve continues instead of forking).")
    p.add_argument("--no-under-penalty", action="store_true",
                   help="Disable the under-bullet (collapse) penalty. Default "
                        "ON in config — defends against single-mega-bullet "
                        "reward hacks (see config.py:length_penalty_under_*). "
                        "Use this flag for A/B against the splitter-only fix.")
    p.add_argument("--under-lambda", type=float, default=None,
                   help="Override cfg['length_penalty_under_lambda'] (default "
                        "0.15). Higher → harsher punishment for reports that "
                        "collapse below half the GT bullet count.")
    p.add_argument("--prioritized-sampling", action="store_true",
                   help="Enable prioritized sampling (historical notes §1.8). Replaces "
                        "ShardGroupedSampler with a weighted-multinomial sampler "
                        "where prompt weights are 1/(running_mean_reward + eps), "
                        "clamped at prioritization_max_weight. Tuned by the "
                        "prioritization_* config knobs. Mutually exclusive "
                        "with --overfit-size and --hard-ids-json.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = dict(RL_CONFIG)
    for k_cli, k_cfg in [
        ("sft_init_from", "sft_init_from"),
        ("pref_loss", "pref_loss"),
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("n_candidates", "n_candidates"),
        ("lr", "lr"),
        ("judge_model", "judge_model"),
        ("checkpoint_dir", "checkpoint_dir"),
        ("precomputed_dir", "precomputed_dir"),
        ("dataset_paths", "dataset_paths"),
        ("valid_record_ids", "valid_record_ids"),
        ("wandb_run_name", "wandb_run_name"),
        ("max_opt_steps", "max_opt_steps"),
        ("overfit_size", "overfit_size"),
        ("eval_every", "eval_every"),
        ("kl_estimator", "kl_estimator"),
        ("gen_max_new_tokens", "gen_max_new_tokens"),
        ("kl_coef", "kl_coef"),
        ("ds_min_std", "dynamic_sampling_min_std"),
        ("ds_max_mean", "dynamic_sampling_max_mean"),
        ("grad_accum", "grad_accum"),
        ("hard_ids_json", "hard_ids_json"),
        ("resume", "resume_from"),
        ("under_lambda", "length_penalty_under_lambda"),
    ]:
        v = getattr(args, k_cli, None)
        if v is not None:
            cfg[k_cfg] = v
    if getattr(args, "no_judge_log", False):
        cfg["judge_log_enabled"] = False
    if getattr(args, "prioritized_sampling", False):
        cfg["prioritized_sampling"] = True
    if getattr(args, "no_under_penalty", False):
        cfg["length_penalty_under_enabled"] = False
    train(cfg)
