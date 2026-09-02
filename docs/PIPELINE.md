# Pipeline walkthrough

End-to-end path from raw instruction data to a released PRISM monitor.
Storage roots come from `.env`: `PRISM_DATA_DIR` (datasets + activation
shards) and `PRISM_CKPT_DIR` (checkpoints).

## 0. Serve the target model (data generation only)

Oracle data generation talks to any OpenAI-compatible endpoint serving the
target model (default `http://localhost:8089/v1`), e.g.
`vllm serve Qwen/Qwen3.5-9B --port 8089`.

## 1. Oracle dataset — download the release, or generate your own

The released records train every recipe directly:

```bash
uv run python scripts/download_dataset.py   # → $PRISM_DATA_DIR/prompt-only (SHA-verified)
uv run python scripts/check_dataset.py --dataset-dir $PRISM_DATA_DIR/prompt-only
```

Alternatively, generate a fresh dataset under `$PRISM_DATA_DIR/prompt-only/`,
then clean it (sampling and judge filtering are not bitwise deterministic, so
a regenerated dataset approximates the released one):

```bash
scripts/generate_dataset.sh     # → prompt-only/jsonl/*.jsonl
scripts/clean_dataset.sh        # filter + valid_record_ids.json mask
```

Cleaning = the rules/LLM-judge filter plus a second rules layer
(≤6-bullet cap, template-leak, word-fragmentation) captured in
`valid_record_ids.json`. The mask lives next to `jsonl/`; the activation
precompute copies it into the cache directory and the on-the-fly loader
picks it up from next to the JSONL files, so both training paths apply it
automatically (pass `--valid-record-ids` to override).

Each record: `{id, source_dataset, prompt, response, retrieval_prompt,
instruction_set, metadata}` where `prompt` is an instruction-rich user prompt
(IFEval / IF-multi-constraints / UltraChat derivatives), `response` the
target model's answer, and `instruction_set` the oracle label — the bullet list
of instructions the model was given ("prompt-only": labeled from the
prompt alone). The filter applies rule gates plus an LLM judge and emits a
`valid_record_ids.json` mask honoured downstream.

## 2. Activations — precomputed cache (default) or on-the-fly

```bash
# cache: built automatically the first time recipes/sft_<model>.sh (or grpo_<model>.sh) runs;
# standalone: uv run python -m prism.activations.extract ...
# on-the-fly instead: PRISM_ON_THE_FLY=1 recipes/sft_<model>.sh / recipes/grpo_<model>.sh
```

Both paths run the frozen target model over `prompt + response` and
take the residual stream at the profile's hook layer for the last ≤128
response tokens.

- **Precomputed cache** (`prism.activations.extract`, the default): the
  activations are extracted once and written as sharded safetensors, with
  the train/val/test split frozen at extraction time (seed 42, 0.1/0.1;
  paraphrase groups never straddle splits) and the `valid_record_ids.json`
  mask copied alongside. Training reads these tensors from disk and avoids
  the activation-extraction forward for each batch. SFT and GRPO still load
  the target model because it serves as the LoRA-adapted decoder. The cache
  improves training throughput at the cost of additional storage. All results
  in the paper and every released checkpoint were trained from the cache.
- **On-the-fly** (`--dataset-paths` on both `prism.sft.train` and
  `prism.rl.train`; `PRISM_ON_THE_FLY=1` in the recipes): the trainer reads
  the JSONL files, applies the same mask and the same split function with the
  same parameters (`split_*` keys in `src/prism/sft/config.py`). Train and
  validation membership therefore match a cache built from the same files.
  Each batch is passed through the frozen base model with LoRA adapters
  disabled, and the forward stops at the hook layer. The main additional
  model computation is this no-grad partial forward; GRPO records it as
  `timing/extract_s` in W&B. This mode avoids the activation cache and is
  useful when changing hook layers during development. It differs from the
  cache path in training order (plain shuffle rather than shard-grouped
  shuffle) and validation order. For Qwen3.5, it also uses the training-time
  model class rather than the extractor's model class, so cached and in-loop
  activations differ slightly (`scripts/check_onthefly_parity.py` measures
  the difference).

`scripts/check_onthefly_parity.py --precomputed-dir <cache> --dataset-paths
<the cache's JSONLs>` verifies the two paths against each other: identical
split membership, identical decoder prefix, and the activation gap (max/mean
|Δ|, cosine) for a sample of records. Split identity holds for caches built by
this repository's extractor from the same files.

## 3. SFT — `prism.sft.train`

```bash
recipes/sft_<model>.sh
```

The monitor = a linear projection (`hidden → hidden`) mapping frozen
activations into the target model's own embedding space, prepended as soft
tokens, plus LoRA (r=32, α=64, 7 proj modules) on the target model, trained
with cross-entropy to emit `instruction_set`. `skip_prompt_b=True` — the
monitor decodes from activations alone, no text prompt at train time.
Best checkpoint by val loss.

## 4. GRPO — `prism.rl.train`

```bash
scripts/serve_judge.sh             # once: vLLM judge endpoint (Gemma-4-31B class)
recipes/grpo_<model>.sh
```

For each prompt, sample N candidate reports (T=1.2), score each with the
LLM judge against the rubric (docs/RUBRIC.md): reward =
`1.0·coverage − 0.4·hallucination_rate − length_penalty`. Group-relative
advantages (GRPO) + k3 KL (coef 0.05) to the frozen SFT reference;
trainable surface = projection + LoRA, matching SFT. DAPO-style dynamic
sampling drops all-tied / near-ceiling groups; the transfer runs add
prioritized sampling and an under-length collapse penalty
(docs/RECIPES.md). Best checkpoint by val judge reward.

Every run writes per-candidate `judge_traces.jsonl` next to its checkpoints
— inspect with `scripts/analyze_judge_traces.py`; an optional hard-example
curriculum can be built from them via `prism.rl.build_hard_ids` and fed back
with `--hard-ids-json`.

## 5. Export — `scripts/export_checkpoint.py`

```bash
uv run python scripts/export_checkpoint.py $PRISM_CKPT_DIR/<run>/best.pt
```

Strips optimizer/scheduler/tracker state, sanitizes the embedded config,
and emits `prism-{target}-{method}.pt` in the exact format
[prism-eval](https://github.com/Offensive-AI-Lab/prism-eval) loads
(docs/CHECKPOINT_FORMAT.md). Evaluation itself — the 1000-record
adversarial suite, judges, baselines — lives entirely in prism-eval.

## Environment notes

- `transformers>=5.3,<6`: the Qwen3.5 profile patches
  `Qwen3_5Model.compute_3d_position_ids`, and gemma-2 must load with
  `attn_implementation="eager"` — sdpa silently drops the attention/logit
  softcapping and produces wrong activations. Both are pinned in the
  target-model profiles.
- The trainers assume a single ~95 GB GPU (gradient checkpointing on, k3 KL);
  see the memory notes in `src/prism/rl/config.py`.
