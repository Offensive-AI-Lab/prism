# Known issues and documented quirks

This repo ships the code **as it behaved when the released checkpoints were
trained**. Bugs whose fixes would change training/reward semantics were
deliberately left in place and documented here instead; crashes, silent
data-corruption, and portability bugs *were* fixed (they cannot affect the
released checkpoints, which were trained on already-generated data).

## Reward / judge (affects GRPO training — left as released)

- **Same judge family trains, validates, and selects.** The GRPO reward, the
  val metric, and `best.pt` selection all use the same judge model + rubric.
  Any judge idiosyncrasy the policy exploits shows up as val improvement.
  Mitigations that exist outside this repo: human κ-calibration of the judge
  (docs/CALIBRATION.md) and a family-disjoint calibrated judge re-scoring of
  all released systems (the paper's judge-swap validation; ranking preserved).
- **Judge-error groups get a negative reward, not zero.** If a judge call
  errors after retries, the candidate scores parse as zero bullets → the
  under-length penalty fires (reward ≈ −0.15·0.5·n_gt instead of 0), the
  batch still trains, and the prioritized-sampling EMA treats the record as
  hard (over-sampling it up to the weight clamp). Transient judge outages
  therefore perturb the curriculum. The released runs trained with this
  behavior.
- **Truncated/missing judge hallucination output defaults to 0.0**
  ("grounded") with judge `max_tokens=200` — a reward-inflation path under
  judge verbosity.
- **Reward is gameable at the margins**: bullets grounded in the *response*
  text (not just prompt/GT) earn credit, and vague type-level bullets earn
  0.5 recall; candidate bullets are pasted into the judge prompt without
  delimiter hardening. The observed val plateaus are consistent with a
  partially-hacked equilibrium; the paper's human-gold calibration and
  judge-swap results bound the practical impact.
- **Sampling/scoring mismatch (biased gradient estimator).** Rollouts sample
  at T=1.2/top_p=0.95 but log-probs are scored on raw T=1.0 logits (no
  temperature scaling, no top-p correction), plus one-step off-policy
  staleness in the async pipeline with no importance correction. Standard
  RLHF stacks temperature-scale here; the released runs did not.

## Metrics (reporting caveats)

- **SFT "token accuracy" is teacher-forced** next-token argmax over labels
  that include chat-template scaffolding — it overstates free-running
  reconstruction quality. Use the eval suite in prism-eval for real numbers.
- **SFT sample tables / BERTScore use right-padded batched generation**
  (HF decoder-only generation wants left padding), so those logged artifacts
  are unreliable for short samples; teacher-forced val loss/accuracy — which
  select `best.pt` — are unaffected.
- Val loss/acc are means of per-batch means (short final batches slightly
  overweighted); val subset for RL is the first ~500 records of an
  unshuffled shard-ordered split (cache path) or of the deterministic
  split order (on-the-fly path) — the same membership, different order.

## Data pipeline (design caveats that persist after the bug-fix pass)

- **Label circularity:** the target model is also the labeler (and the
  filter judge runs on the same served model): `response_b` is the target
  model's own restatement of `prompt_a`'s instructions. When the target
  model misparses a prompt, activations and label encode the same misparse. The
  paper bounds this with human-audited gold and disjoint-judge scoring.
- **IFEval appears as a training source** — don't evaluate these models on
  IFEval-derived benchmarks.
- The judge-filter mask (`valid_record_ids.json`) gates train *and* eval
  splits of the precompute — eval is selected for label consistency.
- **The released runs' train/val/test membership is not reproducible from
  the shipped extractor.** The activation cache behind the released
  checkpoints was built by an earlier version of the extraction script; the
  shipped `prism.activations.extract` (and the on-the-fly loader, which uses
  the same split function) yields a different — equally valid — split of the
  same records for every seed. Regenerated data does not bit-reproduce the
  original JSONLs either (see README), so this only matters if you obtain the
  original files.
- Response token counts can drift between `transformers` versions for a few
  records with non-Latin scripts (tokenizer changes), so a cache built with an
  older version may hold a few tokens more or fewer than on-the-fly
  extraction produces today; `scripts/check_onthefly_parity.py` reports such
  records.
- The labeling judge historically ran with `enable_thinking=True` and a
  verdict regex that could match inside the reasoning text (fixed going
  forward; the released mask was produced by the old parser).

## Environment pins

- `transformers>=5.3,<6`: the Qwen3.5 profile monkeypatches
  `Qwen3_5Model.compute_3d_position_ids` (cached-rope batch-shrink bug in a
  text-only workflow), and gemma-2 must load with `attn_implementation=
  "eager"` — sdpa silently drops attn/final-logit softcapping and produces
  wrong activations. Both are version-sensitive.
- Qwen activation extraction loads `AutoModelForCausalLM` while training
  loads `AutoModelForImageTextToText` (deliberate, matches the released
  precompute); the profiles pin this per model. Consequently on-the-fly
  extraction (`PRISM_ON_THE_FLY=1`, which uses the training-time class) does
  not reproduce the released Qwen cache bit-for-bit; gemma-2 and Ministral
  use one class for both. `scripts/check_onthefly_parity.py` quantifies the
  gap against a cache.
- RL trainers assume a single ~95 GB GPU (gradient checkpointing on, k3 KL);
  see the memory notes in `src/prism/rl/config.py`.
