"""
train.py — Activation-conditioned decoder finetuning.

Pipeline per batch:
  1. Disable LoRA → run the target model on prompt+response → extract activations at the hook layer
  2. Project activations into the decoder embedding space (soft tokens)
  3. Run projection layer → norm-match to decoder embedding scale
  4. Enable LoRA → build decoder input: [soft_tokens | retrieval_prompt embeds | instruction_set embeds]
  5. Teacher-force instruction_set with CE loss, backprop through LoRA + projection

Run:
    python -m prism.sft.train
    python -m prism.sft.train --resume checkpoints/sft-default/best.pt
"""

import argparse
import logging
import os
import random
import time

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from prism.sft.config import FINETUNE_CONFIG
from prism.sft.data import build_collate_fn
from prism.activations import RecordDataset, load_split_records
from prism.sft.model import (
    ActivationProjection,
    register_hook,
    extract_activations,
    compute_target_norm,
    norm_match,
)
from prism.common.env import wandb_mode, load_env
from prism.common.schedule import make_lr_lambda

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_env()


# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────

def _clear_rope_deltas(model: nn.Module):
    """Clear stale rope_deltas cached by Qwen3.5 from a previous forward pass.

    When using precomputed activations we skip the activation-extraction
    forward pass that would normally reset this tensor, so it must be
    cleared explicitly to avoid shape mismatches in compute_3d_position_ids.
    """
    for module in model.modules():
        if getattr(module, "rope_deltas", None) is not None:
            module.rope_deltas = None


def train_step(
    batch: dict,
    target_model,
    projection,
    act_store: dict,
    target_norm: torch.Tensor,
    cfg: dict,
    device: str,
    source_map: dict,
) -> tuple[torch.Tensor, dict]:
    """
    One forward pass: extract activations → project → decode.

    Returns:
        loss: scalar tensor (differentiable)
        log_dict: per-dataset losses and metrics (detached)
    """
    # ── 1. Extract activations (LoRA OFF, no gradients) ──────────────────
    if "precomputed_acts" in batch:
        hook_layer = cfg.get("hook_layer", 16)
        act_padded = batch["precomputed_acts"][hook_layer].to(device)
        act_mask = batch["act_masks"].to(device)
    else:
        target_model.disable_adapter_layers()

        act_padded, act_mask = extract_activations(
            target_model=target_model,
            act_store=act_store,
            input_ids=batch["a_input_ids"].to(device),
            attention_mask=batch["a_attention_mask"].to(device),
            prompt_only_lens=batch["a_prompt_only_lens"],
            response_token_counts=batch["a_response_token_counts"],
            max_act_tokens=cfg["max_act_tokens"],
        )

    # ── 2. Soft tokens = raw activations ─────────────────────────────────
    use_proj = cfg.get("_use_projection", True)
    soft = act_padded

    # ── 3. Project + norm match ──────────────────────────────────────────
    if "precomputed_acts" not in batch:
        target_model.enable_adapter_layers()

    if use_proj:
        soft = projection(soft)
    soft = norm_match(soft, target_norm)
    soft = soft * act_mask.unsqueeze(-1).to(soft.dtype)

    # ── 4. Build decoder input ───────────────────────────────────────────
    chat_ids = batch["chat_input_ids"].to(device)
    chat_mask = batch["chat_attention_mask"].to(device)
    chat_lengths = batch["chat_lengths"]
    answer_starts = batch["answer_starts"]
    source_ids = batch["source_ids"]

    B = chat_ids.shape[0]

    chat_embeds = target_model.get_input_embeddings()(chat_ids)

    combined_embeds = []
    combined_masks = []
    combined_labels = []

    for b in range(B):
        act_len_b = int(act_mask[b].sum().item())
        chat_len_b = int(chat_lengths[b].item())
        ans_start_b = int(answer_starts[b].item())

        soft_b = soft[b, :act_len_b, :]           # [act_len, D]
        chat_emb_b = chat_embeds[b, :chat_len_b]  # [chat_len, D]
        chat_ids_b = chat_ids[b, :chat_len_b]     # [chat_len]

        # Concatenate: [soft_tokens | retrieval_prompt + instruction_set]
        seq_emb = torch.cat([soft_b, chat_emb_b], dim=0)
        seq_mask = torch.ones(seq_emb.size(0), device=device, dtype=chat_mask.dtype)

        # Labels: -100 on soft tokens and retrieval_prompt, actual IDs on instruction_set
        seq_labels = torch.full(
            (seq_emb.size(0),), -100, device=device, dtype=torch.long
        )
        if ans_start_b < chat_len_b:
            seq_labels[act_len_b + ans_start_b : act_len_b + chat_len_b] = \
                chat_ids_b[ans_start_b:chat_len_b]

        combined_embeds.append(seq_emb)
        combined_masks.append(seq_mask)
        combined_labels.append(seq_labels)

    # Pad to batch max
    inputs_embeds = nn.utils.rnn.pad_sequence(
        combined_embeds, batch_first=True, padding_value=0.0
    )
    full_mask = nn.utils.rnn.pad_sequence(
        combined_masks, batch_first=True, padding_value=0
    )
    target_labels = nn.utils.rnn.pad_sequence(
        combined_labels, batch_first=True, padding_value=-100
    )

    # ── 5. Forward through decoder (LoRA ON) ─────────────────────────────
    if "precomputed_acts" in batch:
        _clear_rope_deltas(target_model)
    outputs = target_model(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
        labels=target_labels,
    )
    loss = outputs.loss

    # ── 6. Per-dataset loss logging ──────────────────────────────────────
    log_dict = {}
    with torch.no_grad():
        # Per-token losses for per-dataset breakdown (logging only).
        # Computed sample-by-sample rather than as one [B*seq, vocab]
        # cross_entropy: with a ~250k-token vocab that single fp32 softmax is
        # many GB in one allocation (it was the SFT OOM point on large-vocab
        # targets), whereas the per-sample form frees each [seq, vocab] softmax
        # as it goes. Numerically identical, strictly lower peak memory.
        logits = outputs.logits                       # [B, seq, vocab]
        shift_labels = target_labels[..., 1:]          # [B, seq-1]
        per_token_loss = torch.stack([
            nn.functional.cross_entropy(
                logits[b, :-1, :], shift_labels[b], reduction="none"
            )
            for b in range(B)
        ], dim=0)                                      # [B, seq-1]

        # Token accuracy
        preds = logits[:, :-1, :].argmax(dim=-1)
        label_mask = shift_labels != -100
        correct = (preds == shift_labels) & label_mask
        if label_mask.any():
            log_dict["metrics/token_acc"] = (
                correct.sum().float() / label_mask.sum().float()
            ).item()

        # Per-dataset loss
        inv_source_map = {v: k for k, v in source_map.items()}
        for ds_id in inv_source_map:
            ds_mask = (source_ids == ds_id)  # [B]
            if not ds_mask.any():
                continue
            # Get per-token losses for samples from this dataset
            ds_token_losses = per_token_loss[ds_mask]  # [n_ds, seq-1]
            ds_label_mask = label_mask[ds_mask]         # [n_ds, seq-1]
            if ds_label_mask.any():
                ds_loss = ds_token_losses[ds_label_mask].mean().item()
                log_dict[f"loss/{inv_source_map[ds_id]}"] = ds_loss

    return loss, log_dict


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    val_loader,
    target_model,
    projection,
    act_store: dict,
    target_norm: torch.Tensor,
    tokenizer,
    cfg: dict,
    device: str,
    source_map: dict,
) -> dict:
    """Run validation, return averaged metrics + WandB sample table."""
    use_proj = cfg.get("_use_projection", True)
    if use_proj:
        projection.eval()

    total_loss = 0.0
    total_acc = 0.0
    ds_loss_sums = {v: 0.0 for v in source_map.values()}
    ds_counts = {v: 0 for v in source_map.values()}
    n_batches = 0
    n_samples = 0

    log_sample_count = cfg.get("log_sample_count", 5)
    samples_logged = 0
    table_rows: list[tuple] = []  # deferred — built into wandb.Table after BERTScore

    bert_score_n = cfg.get("bert_score_samples", 0)
    bs_preds: list[str] = []
    bs_refs: list[str] = []

    for batch in val_loader:
        if n_samples >= cfg["eval_samples"]:
            break

        # Extract or load precomputed activations
        if "precomputed_acts" in batch:
            hook_layer = cfg.get("hook_layer", 16)
            act_padded = batch["precomputed_acts"][hook_layer].to(device)
            act_mask = batch["act_masks"].to(device)
        else:
            target_model.disable_adapter_layers()
            act_padded, act_mask = extract_activations(
                target_model=target_model,
                act_store=act_store,
                input_ids=batch["a_input_ids"].to(device),
                attention_mask=batch["a_attention_mask"].to(device),
                prompt_only_lens=batch["a_prompt_only_lens"],
                response_token_counts=batch["a_response_token_counts"],
                max_act_tokens=cfg["max_act_tokens"],
            )

        soft = act_padded

        if "precomputed_acts" not in batch:
            target_model.enable_adapter_layers()

        if use_proj:
            soft = projection(soft)
        soft = norm_match(soft, target_norm)
        soft = soft * act_mask.unsqueeze(-1).to(soft.dtype)

        chat_ids = batch["chat_input_ids"].to(device)
        chat_mask = batch["chat_attention_mask"].to(device)
        chat_lengths = batch["chat_lengths"]
        answer_starts = batch["answer_starts"]
        source_ids = batch["source_ids"]
        ground_truths = batch["instruction_sets"]
        prompts = batch["prompts"]
        responses = batch["responses"]
        B = chat_ids.shape[0]

        chat_embeds = target_model.get_input_embeddings()(chat_ids)

        combined_embeds = []
        combined_masks = []
        combined_labels = []
        prefix_embeds_list = []
        prefix_masks_list = []

        for b in range(B):
            act_len_b = int(act_mask[b].sum().item())
            chat_len_b = int(chat_lengths[b].item())
            ans_start_b = int(answer_starts[b].item())

            soft_b = soft[b, :act_len_b, :]
            chat_emb_b = chat_embeds[b, :chat_len_b]
            chat_ids_b = chat_ids[b, :chat_len_b]

            seq_emb = torch.cat([soft_b, chat_emb_b], dim=0)
            seq_mask = torch.ones(seq_emb.size(0), device=device, dtype=chat_mask.dtype)
            seq_labels = torch.full(
                (seq_emb.size(0),), -100, device=device, dtype=torch.long
            )
            if ans_start_b < chat_len_b:
                seq_labels[act_len_b + ans_start_b : act_len_b + chat_len_b] = \
                    chat_ids_b[ans_start_b:chat_len_b]

            combined_embeds.append(seq_emb)
            combined_masks.append(seq_mask)
            combined_labels.append(seq_labels)

            # Build prefix for generation: [soft_tokens | prompt_b_before_answer]
            # When skip_prompt_b, ans_start_b=0 so prefix is just soft_tokens
            pref_end_b = min(ans_start_b, chat_len_b)
            if pref_end_b > 0:
                pref_emb = torch.cat([soft_b, chat_emb_b[:pref_end_b]], dim=0)
            else:
                pref_emb = soft_b
            pref_mask = torch.ones(pref_emb.size(0), device=device, dtype=chat_mask.dtype)
            prefix_embeds_list.append(pref_emb)
            prefix_masks_list.append(pref_mask)

        inputs_embeds = nn.utils.rnn.pad_sequence(
            combined_embeds, batch_first=True, padding_value=0.0
        )
        full_mask = nn.utils.rnn.pad_sequence(
            combined_masks, batch_first=True, padding_value=0
        )
        target_labels = nn.utils.rnn.pad_sequence(
            combined_labels, batch_first=True, padding_value=-100
        )

        if "precomputed_acts" in batch:
            _clear_rope_deltas(target_model)
        outputs = target_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=target_labels,
        )

        total_loss += outputs.loss.item()

        # Token accuracy
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = target_labels[..., 1:].contiguous()
        preds_tok = shift_logits.argmax(dim=-1)
        label_mask = shift_labels != -100
        correct = (preds_tok == shift_labels) & label_mask
        if label_mask.any():
            total_acc += (correct.sum().float() / label_mask.sum().float()).item()

        # Per-dataset
        per_token_loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(B, -1)

        inv_source_map = {v: k for k, v in source_map.items()}
        for ds_id in inv_source_map:
            ds_batch_mask = (source_ids == ds_id)
            if not ds_batch_mask.any():
                continue
            ds_token_losses = per_token_loss[ds_batch_mask]
            ds_label_mask = label_mask[ds_batch_mask]
            if ds_label_mask.any():
                ds_loss_sums[ds_id] += ds_token_losses[ds_label_mask].mean().item()
                ds_counts[ds_id] += 1

        # ── Generate samples for logging + BERTScore ─────────────────────
        need_gen = samples_logged < log_sample_count or len(bs_preds) < bert_score_n
        if need_gen:
            prefix_embeds = nn.utils.rnn.pad_sequence(
                prefix_embeds_list, batch_first=True, padding_value=0.0
            )
            prefix_mask = nn.utils.rnn.pad_sequence(
                prefix_masks_list, batch_first=True, padding_value=0
            )

            gen_ids = target_model.generate(
                inputs_embeds=prefix_embeds,
                attention_mask=prefix_mask,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=cfg.get("_pad_token_id") or tokenizer.eos_token_id,
                eos_token_id=cfg.get("_gen_eos_id") or tokenizer.eos_token_id,
                use_cache=True,
            )

            gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            for b in range(B):
                if samples_logged < log_sample_count:
                    table_rows.append((
                        inv_source_map.get(int(source_ids[b].item()), "unknown"),
                        prompts[b][:300],
                        responses[b][:500],
                        ground_truths[b][:500],
                        gen_texts[b][:500],
                    ))
                    samples_logged += 1
                if len(bs_preds) < bert_score_n:
                    bs_preds.append(gen_texts[b])
                    bs_refs.append(ground_truths[b])

        n_batches += 1
        n_samples += B

    if use_proj:
        projection.train()

    # ── BERTScore ─────────────────────────────────────────────────────
    per_sample_f1: list[float] = []
    if bs_preds:
        try:
            from bert_score import score as bert_score_fn
            # bert_score chokes on empty strings — substitute a placeholder
            safe_preds = [p if p.strip() else "[empty]" for p in bs_preds]
            safe_refs  = [r if r.strip() else "[empty]" for r in bs_refs]
            _, _, F1 = bert_score_fn(
                safe_preds, safe_refs, lang="en", device="cpu", verbose=False,
            )
            per_sample_f1 = F1.tolist()
        except ImportError:
            logger.warning("bert-score not installed, skipping BERTScore")

    # ── Build wandb sample table (with BERTScore column if available) ─
    has_bs = len(per_sample_f1) >= len(table_rows)
    columns = ["source", "prompt", "response", "ground_truth", "generated"]
    if has_bs:
        columns.append("bert_score_f1")
    sample_table = wandb.Table(columns=columns)
    for i, row in enumerate(table_rows):
        if has_bs:
            sample_table.add_data(*row, round(per_sample_f1[i], 4))
        else:
            sample_table.add_data(*row)

    result = {
        "val/loss": total_loss / max(n_batches, 1),
        "val/token_acc": total_acc / max(n_batches, 1),
        "val/samples": sample_table,
    }
    if per_sample_f1:
        result["val/bert_score_f1"] = sum(per_sample_f1) / len(per_sample_f1)

    inv_source_map = {v: k for k, v in source_map.items()}
    for ds_id, ds_name in inv_source_map.items():
        if ds_counts[ds_id] > 0:
            result[f"val/loss_{ds_name}"] = ds_loss_sums[ds_id] / ds_counts[ds_id]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    projection, target_model, opt_step, best_val_loss, path, cfg,
    optimizer=None, scheduler=None,
):
    from peft import get_peft_model_state_dict
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ck = {
        "projection_state": projection.state_dict() if projection is not None else None,
        "lora_state": get_peft_model_state_dict(target_model),
        "opt_step": opt_step,
        "best_val_loss": best_val_loss,
        "config": cfg,
    }
    if optimizer is not None:
        ck["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        ck["scheduler_state"] = scheduler.state_dict()
    torch.save(ck, path)
    logger.info(f"Checkpoint saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict):
    if not cfg.get("dataset_paths") and not cfg.get("precomputed_dir"):
        raise SystemExit(
            "No training data configured: cfg['dataset_paths'] is empty and "
            "cfg['precomputed_dir'] is None.\n"
            "Either:\n"
            "  - set PRISM_PRECOMPUTED_DIR to a directory of precomputed activation "
            "shards (produced by `python -m prism.activations`), or\n"
            "  - pass --dataset-paths <jsonl>... for on-the-fly activation extraction."
        )
    # Fail before the target model is loaded (minutes + tens of GB) on paths
    # that do not exist.
    import os as _os
    missing = [p for p in (cfg.get("dataset_paths") or []) if not _os.path.exists(p)]
    if missing:
        raise SystemExit(f"dataset_paths not found: {missing}")
    if cfg.get("precomputed_dir") and not _os.path.exists(_os.path.join(cfg["precomputed_dir"], "manifest.json")):
        raise SystemExit(f"precomputed_dir has no manifest.json: {cfg['precomputed_dir']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])

    # ── 1. Load target model ────────────────────────────────────────────
    from peft import LoraConfig, get_peft_model, TaskType
    from prism.target_models import load_target_model, resolve_gen_eos_id

    logger.info(f"Loading {cfg['model_id']} …")
    t0 = time.time()

    target_model, tokenizer, _profile = load_target_model(device)
    cfg["_gen_eos_id"] = resolve_gen_eos_id(_profile, tokenizer)
    cfg["_pad_token_id"] = tokenizer.pad_token_id

    target_model.config.use_cache = False
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad_(False)

    logger.info(f"Target model loaded in {time.time()-t0:.1f}s")

    # ── 2. Apply LoRA ────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
    )
    target_model = get_peft_model(target_model, lora_config)
    target_model.print_trainable_parameters()

    # ── 3. Hook for activation extraction ────────────────────────────────
    use_precomputed = bool(cfg.get("precomputed_dir"))
    base_model = target_model.base_model.model
    act_store = {}
    hook_handle = None
    if not use_precomputed:
        act_store, hook_handle = register_hook(base_model, cfg["hook_layer"])
    else:
        logger.info("Using precomputed activations from %s — skipping activation hook", cfg["precomputed_dir"])

    # ── 4. Resolve projection ────────────────────────────────────────────
    # The monitor feeds raw hooked activations straight into the decoder
    # embedding space: activations → linear projection → soft tokens.
    use_projection = cfg.get("use_projection")
    if use_projection is None:
        use_projection = True
    cfg["_use_projection"] = use_projection  # pass resolved flag to train_step/validate

    projection = None
    if use_projection:
        projection = ActivationProjection(dim=cfg["projection_dim"]).to(device, dtype=torch.bfloat16)
        logger.info(f"Projection: {sum(p.numel() for p in projection.parameters()):,} params (trainable)")
    else:
        logger.info("Projection layer SKIPPED (raw activations fed directly to the decoder)")

    # ── 6. Target norm ───────────────────────────────────────────────────
    target_norm = compute_target_norm(target_model).to(device)
    logger.info(f"Target embedding norm: {target_norm.item():.4f}")

    # ── 7. Data ──────────────────────────────────────────────────────────
    if use_precomputed:
        from prism.activations import PrecomputedActivationDataset, ShardGroupedSampler, build_precomputed_collate_fn

        hook_layer = cfg.get("hook_layer", 16)
        logger.info("Loading precomputed datasets from %s (layer %d)", cfg["precomputed_dir"], hook_layer)
        train_ds = PrecomputedActivationDataset(cfg["precomputed_dir"], split="train", layers=[hook_layer])
        val_ds = PrecomputedActivationDataset(cfg["precomputed_dir"], split="val", layers=[hook_layer])
        test_ds = PrecomputedActivationDataset(cfg["precomputed_dir"], split="test", layers=[hook_layer]) if cfg.get("run_test_eval", False) else None
        source_map = train_ds.source_map
        # Use full val split when precomputed — unless explicitly overridden (e.g. sweeps)
        if cfg.get("_eval_samples_explicit") is None:
            cfg["eval_samples"] = len(val_ds)

        train_collate = build_precomputed_collate_fn(tokenizer, cfg, source_map, layers=[hook_layer], training=True)
        val_collate = build_precomputed_collate_fn(tokenizer, cfg, source_map, layers=[hook_layer], training=False)

        train_sampler = ShardGroupedSampler(train_ds, shuffle=True, seed=cfg["seed"])
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            sampler=train_sampler,
            collate_fn=train_collate,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=4,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            collate_fn=val_collate,
            num_workers=2,
            prefetch_factor=4,
        )
        if test_ds is not None:
            test_loader = DataLoader(
                test_ds,
                batch_size=cfg["batch_size"],
                shuffle=False,
                collate_fn=val_collate,
                num_workers=2,
            )
        else:
            test_loader = None
        logger.info(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds) if test_ds else 0:,}")
    else:
        logger.info("Loading datasets for ON-THE-FLY activation extraction …")
        # Same loader / mask / split as the activation cache (prism.activations.records).
        train_records, val_records, test_records_all, source_map = load_split_records(
            cfg, cfg.get("valid_record_ids"),
        )
        test_records_list = test_records_all if cfg.get("run_test_eval", False) else []
        logger.info(f"Train: {len(train_records):,}  Val: {len(val_records):,}  Test: {len(test_records_list):,}")

        train_ds = RecordDataset(train_records, source_map)
        val_ds = RecordDataset(val_records, source_map)

        train_collate = build_collate_fn(tokenizer, cfg, source_map, training=True)
        val_collate = build_collate_fn(tokenizer, cfg, source_map, training=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
            collate_fn=train_collate,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            collate_fn=val_collate,
        )
        if test_records_list:
            test_ds = RecordDataset(test_records_list, source_map)
            test_loader = DataLoader(
                test_ds,
                batch_size=cfg["batch_size"],
                shuffle=False,
                collate_fn=val_collate,
            )
        else:
            test_loader = None

    # ── 8. Optimizer ─────────────────────────────────────────────────────
    lora_params = [p for p in target_model.parameters() if p.requires_grad]
    proj_params = list(projection.parameters()) if use_projection else []

    param_groups = [
        {"params": lora_params, "lr": cfg["lr"], "name": "lora"},
    ]
    if proj_params:
        param_groups.append({"params": proj_params, "lr": cfg["projection_lr"], "name": "projection"})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["weight_decay"])

    n_lora = sum(p.numel() for p in lora_params)
    n_proj = sum(p.numel() for p in proj_params)
    parts = [f"LoRA: {n_lora/1e6:.1f}M"]
    if n_proj > 0:
        parts.append(f"Projection: {n_proj/1e6:.1f}M")
    logger.info(f"Trainable — {'  '.join(parts)}")

    # ── 9. Scheduler ─────────────────────────────────────────────────────
    grad_accum = cfg["grad_accum"]
    steps_per_epoch = max(1, (len(train_loader) + grad_accum - 1) // grad_accum)
    total_opt_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = int(total_opt_steps * cfg["warmup_ratio"])

    lr_lambda = make_lr_lambda(warmup_steps, total_opt_steps, cfg["min_lr_ratio"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    logger.info(
        f"Steps/epoch: {steps_per_epoch}  Total opt steps: {total_opt_steps}  "
        f"Warmup: {warmup_steps}"
    )

    # ── 10. WandB ────────────────────────────────────────────────────────
    # Skip init if already active (e.g. called from sweep agent)
    if wandb.run is None:
        wandb.init(project=cfg["wandb_project"], config=cfg, name=cfg.get("wandb_run_name", "finetune-local"),
                   mode=wandb_mode())
    else:
        wandb.config.update(cfg, allow_val_change=True)

    # ── 11. Resume ───────────────────────────────────────────────────────
    opt_step = 0
    best_val_loss = float("inf")
    bad_evals = 0

    if cfg.get("resume_from"):
        ck = torch.load(cfg["resume_from"], map_location=device)
        if use_projection and ck.get("projection_state") is not None:
            projection.load_state_dict(ck["projection_state"])
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(target_model, ck["lora_state"])
        opt_step = ck.get("opt_step", 0)
        best_val_loss = ck.get("best_val_loss", float("inf"))
        if "optimizer_state" in ck:
            optimizer.load_state_dict(ck["optimizer_state"])
        if "scheduler_state" in ck:
            scheduler.load_state_dict(ck["scheduler_state"])
        logger.info(f"Resumed from {cfg['resume_from']} (opt_step={opt_step})")

    # ── 12. Training loop ────────────────────────────────────────────────
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    best_path = os.path.join(cfg["checkpoint_dir"], "best.pt")
    latest_path = os.path.join(cfg["checkpoint_dir"], "latest.pt")

    global_step = 0
    # For resume: compute how many micro-batches to skip
    resume_skip = opt_step * grad_accum
    target_model.train()
    if use_projection:
        projection.train()

    for epoch in range(cfg["epochs"]):
        logger.info(f"═══ Epoch {epoch+1}/{cfg['epochs']} ═══")
        epoch_loss = 0.0
        epoch_batches = 0

        for batch in train_loader:
            # Skip batches already processed before checkpoint
            if global_step < resume_skip:
                global_step += 1
                continue

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, log_dict = train_step(
                    batch=batch,
                    target_model=target_model,
                    projection=projection,
                    act_store=act_store,
                    target_norm=target_norm,
                    cfg=cfg,
                    device=device,
                    source_map=source_map,
                )
                loss_scaled = loss / grad_accum

            loss_scaled.backward()
            epoch_loss += loss.item()
            epoch_batches += 1
            global_step += 1

            if global_step % grad_accum == 0:
                # Separate gradient clipping per parameter group
                gnorms = {}
                gnorms["lora"] = torch.nn.utils.clip_grad_norm_(
                    lora_params, cfg["grad_clip"]
                ).item()
                if proj_params:
                    gnorms["proj"] = torch.nn.utils.clip_grad_norm_(
                        proj_params, cfg["grad_clip"]
                    ).item()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                # ── Logging ──────────────────────────────────────────
                if opt_step % cfg["log_every"] == 0:
                    lrs = {pg["name"]: pg["lr"] for pg in optimizer.param_groups}
                    wandb_log = {
                        "train/loss": loss.item(),
                        "train/lr_lora": lrs.get("lora", cfg["lr"]),
                        "train/epoch": epoch + (epoch_batches / len(train_loader)),
                        "opt_step": opt_step,
                        **{f"train/{k}": v for k, v in log_dict.items()},
                    }
                    for name, gn in gnorms.items():
                        wandb_log[f"train/grad_norm_{name}"] = gn
                    if "projection" in lrs:
                        wandb_log["train/lr_projection"] = lrs["projection"]
                    wandb.log(wandb_log, step=opt_step)

                    ds_losses_str = "  ".join(
                        f"{k}={v:.4f}" for k, v in log_dict.items()
                        if k.startswith("loss/")
                    )
                    gnorm_str = "  ".join(f"gnorm_{k}={v:.2f}" for k, v in gnorms.items())
                    logger.info(
                        f"[step {opt_step}/{total_opt_steps}] "
                        f"loss={loss.item():.4f}  "
                        f"acc={log_dict.get('metrics/token_acc', 0):.3f}  "
                        f"lr={lrs.get('lora', 0):.2e}  "
                        f"{gnorm_str}  "
                        f"{ds_losses_str}"
                    )

                # ── Validation ───────────────────────────────────────
                if opt_step % cfg["eval_every"] == 0:
                    # Periodic checkpoint so progress survives preemption
                    save_checkpoint(
                        projection, target_model,
                        opt_step, best_val_loss, latest_path, cfg,
                        optimizer, scheduler,
                    )

                if opt_step % cfg["eval_every"] == 0 and len(val_ds) > 0:
                    target_model.eval()
                    val_log = validate(
                        val_loader, target_model, projection,
                        act_store, target_norm, tokenizer, cfg, device,
                        source_map,
                    )
                    target_model.train()

                    val_log["opt_step"] = opt_step
                    wandb.log(val_log, step=opt_step)

                    val_ds_str = "  ".join(
                        f"{k}={v:.4f}" for k, v in val_log.items()
                        if k.startswith("val/loss_")
                    )
                    logger.info(
                        f"  [VAL] loss={val_log['val/loss']:.4f}  "
                        f"acc={val_log.get('val/token_acc', 0):.3f}  "
                        f"{val_ds_str}"
                    )

                    if val_log["val/loss"] < best_val_loss:
                        best_val_loss = val_log["val/loss"]
                        bad_evals = 0
                        save_checkpoint(
                            projection, target_model,
                            opt_step, best_val_loss, best_path, cfg,
                            optimizer, scheduler,
                        )
                        logger.info(f"  ★ New best val loss: {best_val_loss:.5f}")
                    else:
                        bad_evals += 1

                    if bad_evals >= cfg["early_stop_patience"]:
                        logger.info(
                            f"Early stopping: {bad_evals} evals without improvement."
                        )
                        break

        # End of epoch
        if bad_evals >= cfg["early_stop_patience"]:
            break

        save_checkpoint(
            projection, target_model,
            opt_step, best_val_loss, latest_path, cfg,
            optimizer, scheduler,
        )
        logger.info(
            f"Epoch {epoch+1} done. "
            f"Avg loss: {epoch_loss / max(epoch_batches, 1):.4f}"
        )

    # ── Held-out test evaluation ────────────────────────────────────────
    if test_loader is not None and len(test_loader.dataset) > 0:
        logger.info("Running held-out test evaluation …")
        # Load best checkpoint for test eval
        if os.path.exists(best_path):
            ck = torch.load(best_path, map_location=device)
            if use_projection and ck.get("projection_state") is not None:
                projection.load_state_dict(ck["projection_state"])
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(target_model, ck["lora_state"])
            logger.info(f"Loaded best checkpoint for test eval: {best_path}")

        test_cfg = {**cfg, "eval_samples": len(test_loader.dataset), "log_sample_count": 0}
        test_log = validate(
            test_loader, target_model, projection,
            act_store, target_norm, tokenizer, test_cfg, device,
            source_map,
        )
        # Rename val/ → test/ keys for wandb
        test_metrics = {}
        for k, v in test_log.items():
            if isinstance(v, (int, float)):
                test_metrics[k.replace("val/", "test/")] = v
        wandb.log(test_metrics, step=opt_step)
        logger.info(
            f"  [TEST] loss={test_metrics.get('test/loss', 0):.4f}  "
            f"acc={test_metrics.get('test/token_acc', 0):.3f}"
        )

    # ── Done ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Best model saved at: {best_path}")
    logger.info("=" * 60)

    if hook_handle is not None:
        hook_handle.remove()
    wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Finetune activation-conditioned decoder")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint .pt to resume from")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--precomputed-dir", type=str, default=None,
                   help="directory of precomputed activation shards (overrides PRISM_PRECOMPUTED_DIR)")
    p.add_argument("--dataset-paths", nargs="+", default=None,
                   help="Instruction-set JSONL files for ON-THE-FLY activation extraction "
                        "(env: PRISM_DATASET_PATHS). Ignored when a precomputed dir is set.")
    p.add_argument("--valid-record-ids", type=str, default=None,
                   help="valid_record_ids.json mask for --dataset-paths (default: auto-detect "
                        "next to the JSONL files; 'none' to disable).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = dict(FINETUNE_CONFIG)

    if args.resume is not None:     cfg["resume_from"] = args.resume
    if args.epochs is not None:     cfg["epochs"] = args.epochs
    if args.batch_size is not None: cfg["batch_size"] = args.batch_size
    if args.lr is not None:         cfg["lr"] = args.lr
    if args.dataset_paths is not None:
        cfg["dataset_paths"] = args.dataset_paths
    if args.valid_record_ids is not None:
        cfg["valid_record_ids"] = args.valid_record_ids
    if args.precomputed_dir is not None:
        cfg["precomputed_dir"] = args.precomputed_dir

    train(cfg)
