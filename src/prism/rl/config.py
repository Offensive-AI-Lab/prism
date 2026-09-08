"""RL_CONFIG — GRPO on top of the SFT (projection + LoRA) checkpoint.

Inherits from `prism.sft.config.FINETUNE_CONFIG` and adds RL-specific knobs.
The values for `lora_r`, `lora_alpha`, `lora_target_modules`,
`use_projection`, `skip_prompt_b`, `hook_layer`, `max_act_tokens`,
`precomputed_dir` are pinned to match the chosen SFT checkpoint so the policy
adapter loads cleanly.
"""

from __future__ import annotations

import os

from prism.sft.config import FINETUNE_CONFIG

# The SFT checkpoint to initialise RL from. Must carry both lora_state and
# projection_state (trained with use_projection=True).
# Set via the PRISM_SFT_INIT_FROM env var or --sft-init-from; the trainer
# fails fast with a clear error when unset.
SFT_INIT_FROM = os.environ.get("PRISM_SFT_INIT_FROM") or None

RL_CONFIG = {
    **FINETUNE_CONFIG,

    # ─── Pinned to match SFT checkpoint ───────────────────────────────────────
    "use_projection": True,
    "projection_dim": 4096,
    "skip_prompt_b": True,
    "hook_layer": 16,
    "max_act_tokens": 128,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    # ─── RL loss selector ─────────────────────────────────────────────────────
    # "grpo" (recommended), "dpo", or "ipo".
    "pref_loss": "grpo",
    "kl_coef": 0.05,          # GRPO. History:
                              #   0.04 — original; effectively zero under the
                              #     legacy seq-proxy KL (wrong scale).
                              #   0.2  — bumped after the length-norm + token-KL
                              #     fix so the regularizer had real weight on
                              #     the new (per-token, ~0.01 magnitude) scale.
                              #   0.05 — relaxed after early runs
                              #     plateaued at val≈0.78 from step 200 onward
                              #     (std across val passes < per-pass SE → real
                              #     KL-pinned plateau, not noise). 0.05 gives
                              #     the policy 4× more room to drift from ref
                              #     while still preventing collapse.
    "beta": 0.1,              # DPO / IPO
    "min_margin": 0.25,       # DPO / IPO pair filter
    # GRPO loss-shape knobs — defaults match the post-fix recommendation
    # (length-normalised policy logp + exact per-response-token KL). Flip
    # length_normalize_logp to False to reproduce the pre-fix loss for A/B.
    "length_normalize_logp": True,
    # KL estimator selector:
    #   "token_exact" — exact per-token KL over full vocab. Most accurate,
    #                   but the fp32 log_softmax on [B*N, T, V] is the
    #                   dominant memory cost. Caps practical N at ~4.
    #   "k3"          — Schulman's per-sampled-token estimator
    #                   `(r - 1) - log(r)`. Non-negative, low-variance, and
    #                   uses ONLY sampled-token log-probs — never holds full
    #                   vocab logits for KL. ~3-5 GB lighter per loss step,
    #                   enables N=6+ on a 95 GB GPU. Standard in production
    #                   GRPO (TRL, OpenRLHF). Recommended.
    #   "seq_proxy"   — legacy `(pi_logp - ref_logp).mean()`. Kept for A/B;
    #                   can go negative on individual samples.
    "kl_estimator": "token_exact",

    # ─── Trainable surface in RL ─────────────────────────────────────────────
    # Both projection AND LoRA are trainable, matching SFT posture.
    # The ref forward uses a frozen SFT-snapshot of the projection.
    "train_projection_in_rl": True,
    "projection_lr": 5e-6,    # match `lr` by default; tune separately if needed

    # ─── Rollout (candidate generation) ──────────────────────────────────────
    # gen_max_new_tokens trimmed from 256 → 192: calibration reports averaged
    # ~120 tokens with a long tail; 192 leaves comfortable headroom while
    # cutting generation wall time ~25%. Length penalty already handles
    # over-verbosity if the model decides to push past the cap.
    "n_candidates": 4,
    # ─── Dynamic sampling (DAPO-style group filter) ──────────────
    # Per-prompt filters applied AFTER judging, BEFORE the policy/ref forwards.
    # A group whose N candidates have:
    #   - std < dynamic_sampling_min_std       → all-tied / no learning signal.
    #     GRPO advantages collapse to ~0 but the KL term still drags the policy
    #     toward ref → net effect is "free" regularization. Drop it.
    #   - mean > dynamic_sampling_max_mean     → near judge ceiling. Any
    #     below-1.0 candidate gets a huge negative advantage driven by judge
    #     stochasticity (e.g. one bullet flipping 1.0→0.5), pushing the policy
    #     AWAY from near-perfect responses. Drop it.
    # If all groups in a batch are dropped, the whole optimizer step is
    # skipped (no fwd/bwd/opt). Set min_std=0 and max_mean≥2 to disable
    # entirely (legacy behaviour, every group trains).
    "dynamic_sampling_min_std": 0.05,
    "dynamic_sampling_max_mean": 0.95,
    # ─── Prioritized sampling ──────────────────────────
    # Sample prompts with weight inversely proportional to current running-mean
    # reward, so the policy spends rollout budget on prompts it's currently
    # weakest on without fully excluding the easy ones (forgetting risk).
    # Replaces ShardGroupedSampler with PrioritizedSampler when enabled.
    # See prism/rl/sampler.py.
    #
    # Defaults are tuned so:
    #   - hardest prompt is sampled ~5x more often than easiest (max_weight clamp)
    #   - unvisited prompt has weight ~1.67 (neutral)
    #   - EMA over last ~3-4 visits (alpha=0.3)
    #   - first 500 steps use uniform weights (warmup) so the tracker has
    #     observations before priorities kick in
    "prioritized_sampling": False,            # opt-in via cfg or --prioritized-sampling
    "prioritization_epsilon": 0.1,            # weight = 1 / (running_mean + epsilon)
    "prioritization_max_weight": 5.0,         # clamp; prevents one freak-low reward
                                              # from monopolizing the sampler
    "prioritization_default_mean": 0.8,       # for unvisited records.
                                              # History: 0.5 → unvisited weight
                                              # 1.67, which was HIGHER than
                                              # known-easy (~1.0) — unvisited
                                              # records were over-sampled
                                              # relative to known-easy, which
                                              # slowed early curriculum
                                              # convergence. 0.8 →
                                              # unvisited weight 1.11, neutral-
                                              # to-slightly-above-easy.
    "prioritization_alpha": 0.3,              # EMA on new observation
    "prioritization_warmup_steps": 5000,      # uniform weights until step ≥ this.
                                              # History: 500 was way too short
                                              # for a 222k-record dataset — at
                                              # step 500 the tracker had ~1000
                                              # visits, so the weight tensor
                                              # was 99.5% default; priority
                                              # weighting was essentially a
                                              # no-op until coverage saturated
                                              # around step 2500+.
                                              # 5000 lets the tracker visit a
                                              # meaningful fraction (~5%+ with
                                              # multiple visits each) before
                                              # priorities kick in — priority
                                              # then concentrates on prompts
                                              # with statistically reliable
                                              # weight estimates rather than
                                              # first-impression noise.
                                              # Weight-rebuild cadence is
                                              # locked to `eval_every` (rebuilds
                                              # happen inside the val block) —
                                              # there's deliberately no separate
                                              # `rebuild_every` knob because the
                                              # weights would just go stale
                                              # between val passes anyway.
    # Gradient checkpointing on the target model — recompute forward activations during
    # backward instead of storing them. Lets us scale n_candidates past 6 on
    # the 95 GB RTX 6000 Pro. Trade-off: ~30% slower training step. Disable
    # only if you have memory headroom and want max throughput.
    "gradient_checkpointing": True,
    # Chunk sizes for the memory-bounded loss path. Both are
    # mathematically exact (chunked log_softmax + chunked KL accumulation
    # produce the same scalar as the unchunked version, up to fp32 noise).
    # logp_chunk_size=4 caps the fp32 log_softmax peak at [4, T, V] ≈ 2.5 GB
    # at T=1024; kl_chunk_size=256 keeps the KL fp32 peak under ~1 GB.
    # Set to None to disable chunking (only safe at N≤4 on a 95 GB GPU).
    # Dropped 2→1 after grpo-hard-t1.2-ga2-n6-k3 OOM'd at step 6 on the
    # ref forward: hard-ids subset over-represents if_multi_constraints
    # (long-prompt source) → larger T → chunked fp32 cast peak exceeded
    # the available headroom. Pair with PYTORCH_ALLOC_CONF=expandable_segments:True.
    "logp_chunk_size": 1,
    "kl_chunk_size": 256,
    # torch.cuda.empty_cache() between train steps. Defragments the allocator
    # between variable-length candidate batches; cheap (microseconds) and
    # prevents the "94 GB in use, 234 MB unallocated, can't fit next 1 GB"
    # allocator-fragmentation OOM pattern observed in early runs.
    "empty_cache_each_step": True,
    "gen_temperature": 1.2,    # Bumped from 1.0 → 1.2 to widen rollout
                               # diversity → larger group reward spread →
                               # more groups survive the dynamic-sampling
                               # filter. Pair with top_p=0.9 if degenerate
                               # / format-broken candidates appear.
    "gen_top_p": 0.95,
    "gen_max_new_tokens": 192,

    # ─── Judge ────────────────────────────────────────────────────────────────
    # judge_workers raised 8 → 12: local vLLM Gemma endpoint handles concurrent
    # requests comfortably; the extra headroom covers transient retries
    # without blocking the training step.
    "judge_model": os.environ.get("PRISM_JUDGE_MODEL") or "MODEL_NOT_SET",
    "judge_workers": 12,
    "judge_max_retries": 5,

    # ─── Reward formula ───────────────────────────────────────────────────────
    # reward = w_inst * mean_instruction_score
    #        - w_halluc * mean_hallucination_score
    #        - length_penalty
    # mean_instruction_score (recall): mean over GT-bullet scores, higher=better.
    # mean_hallucination_score (precision): mean over ITM-bullet scores, higher=worse.
    # length_penalty: λ * max(0, n_itm_bullets − k · n_gt_bullets), guards against
    # padding ITM reports with extra bullets to farm partial credit.
    #
    # The released weights favor instruction coverage over hallucination; the
    # theoretical maximum is 1.0 for full coverage with no hallucination or
    # length penalty.
    "instruction_weight": 1.0,
    "hallucination_weight": 0.4,
    "length_penalty_enabled": True,
    "length_penalty_k": 1.5,
    "length_penalty_lambda": 0.15,
    # ─── Under-bullet (collapse) penalty ──────────────────────────────────────
    # Defense-in-depth against reward-hacks that bundle multiple claims into
    # one bullet so the per-bullet hallucination axis dilutes to ~0. Observed
    # in an early run: policy learnt to emit
    # the whole report on a single line as "- a - b - c - d", collapsing the
    # judge view to 1 always-grounded claim. The inline-dash splitter in
    # judge.split_instructions catches that exact form; this penalty
    # catches the rest (' ; ', ' / ', single paragraph, etc.) by punishing
    # reports shorter than under_k · n_gt.
    #
    # Defaults — under_k=0.5 fires only when the report has fewer than HALF
    # the GT bullet count, so legitimate merged-bullet responses (rubric §1
    # MERGED instructions) aren't penalised. λ=0.15 matches the overrun
    # weight, so a GT=5 / report=1 collapse costs 0.15·(2.5−1)=0.225 — large
    # enough to outweigh the dropped hallucination penalty (≤0.4 · 1 = 0.4)
    # only at the worst end; tune up if collapse persists.
    "length_penalty_under_enabled": True,
    "length_penalty_under_k": 0.5,
    "length_penalty_under_lambda": 0.15,

    # ─── Init ─────────────────────────────────────────────────────────────────
    "sft_init_from": SFT_INIT_FROM,

    # ─── Training cadence (RL is more expensive per step than SFT) ──────────
    # grad_accum dropped 4 → 1: GRPO's advantage is group-relative within each
    # batch's N candidates, so accumulating 4 micro-steps just averages 4
    # already-baselined gradients — same effect as a slightly lower LR but
    # 4× the wall time. Effective batch (B·N=8) per update is plenty.
    # eval_samples=500 gives SE ≈ 0.012 on val/reward_mean (calibration std
    # 0.263 / √500), tight enough to detect ~0.03+ deltas between consecutive
    # val passes. SFT used 1500 but its val was 12× cheaper per sample
    # (forward-only). At 500 we keep total val overhead to ~30% of run time.
    # log_sample_count=10 matches SFT's pattern; the samples are essentially
    # free once we've generated + judged them.
    "batch_size": 2,
    "val_batch_size": 16,  # Bigger than train batch_size — val is greedy
                           # n_candidates=1 with no autograd / no ref forward,
                           # so generate batches near-linearly. ~6-8× faster
                           # val pass than at batch_size=2.
                           # Drop to 8 if val OOMs on a particularly long
                           # generation batch.
    "grad_accum": 1,
    "epochs": 1,
    "lr": 5e-6,
    "log_every": 5,
    "eval_every": 200,
    "eval_samples": 500,
    # Mid-interval checkpoint cadence — decoupled from eval to bound the
    # work-loss of a crash. Latest.pt is also saved at every eval pass; this
    # adds an extra save every `save_every` train steps. At 763 MB and ~5 s
    # to NFS, save_every=50 is ~0.9% wall-clock overhead. Set save_every >=
    # eval_every to disable the extra saves (back to "val-only saves").
    "save_every": 50,
    "log_sample_count": 10,

    # ─── Checkpoint / wandb ──────────────────────────────────────────────────
    "checkpoint_dir": "./checkpoints/grpo-default",
    "wandb_project": "prism-rl",
    "wandb_entity": None,
    "wandb_run_name": "grpo-default",

    # ─── Run-length cap + judge-trace logging ────────────────────────────────
    # max_opt_steps=2000 (~14h on a ~95 GB GPU) is the bare default; the
    # released recipes pass --max-opt-steps 20000. Set to None to run until
    # epochs are exhausted. judge_log_enabled writes per-candidate JSONL
    # trace to <checkpoint_dir>/judge_traces.jsonl; disable with --no-judge-log.
    "max_opt_steps": 2000,
    "judge_log_enabled": True,

    # ─── Hard-sample subset (optional) ───────────────────────────────────────
    # Path to a JSON produced by prism.rl.build_hard_ids. When set,
    # the train dataset is filtered to only those record_ids before the
    # DataLoader is built. Cuts wall time spent on records the policy is
    # already ceiling-hitting (rollout + judge cost), at the risk of mild
    # forgetting on the easy slice — the build script keeps unobserved records
    # by default to hedge against that. ShardGroupedSampler is disabled in
    # this mode (Subset doesn't expose shard boundaries); falls back to
    # vanilla shuffled DataLoader.
    "hard_ids_json": None,
}

# ─── Target-model overlay ─────────────────────────────────────────────────────
# Re-apply after the RL dict because it re-pins hook_layer / projection_dim /
# lora_target_modules to the SFT recipe (which for gemma2/ministral3 differ from
# Qwen). No-op for Qwen (default). Also honours PRISM_SFT_INIT_FROM etc.
from prism.target_models import apply_profile_overlay as _apply_profile_overlay  # noqa: E402
_apply_profile_overlay(RL_CONFIG)
