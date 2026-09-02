# Data card

This repo ships **no datasets** — only generation code. This card documents
what the pipeline produces, from what sources, and under what licenses.

## Oracle training data (generated, not shipped)

| field | contents |
|---|---|
| `prompt_a` | instruction-rich user prompt, drawn from the sources below |
| `response_a` | target model's answer to `prompt_a` (sampled) |
| `prompt_b` | fixed oracle prompt ("summarize the instructions you were given") |
| `response_b` | oracle label: bullet list of the instructions in `prompt_a` (prompt-only: generated from the prompt alone, temp 0.3) |
| `metadata` | source, paraphrase group id, generation parameters |

### Sources pulled at generation time

| source key | upstream | license notes |
|---|---|---|
| `if_eval` | google/IFEval (HF) | Apache-2.0 |
| `if_multi_constraints` | IF-multi-constraints-style prompts | check the configured HF dataset's card |
| `ultrachat` | HuggingFaceH4/ultrachat_200k (HF) | MIT per its HF dataset card (a filtered derivative of the UltraChat corpus) — verify the card before redistributing derived data |

Generated `response_a`/`response_b` are outputs of the target model
(Qwen3.5-9B for the released runs) — additionally subject to the target
model's license/usage terms. **If you redistribute generated JSONLs, you
inherit all of the above**; that is why this repo ships scripts only.

### Labels

`response_b` is the **prompt-only** oracle label: the target model is shown
`prompt_a` alone and asked to list the instructions it contains (temperature
0.3). The generated records are then cleaned — rules + LLM-judge filter, a
≤6-bullet cap, template-leak and word-fragmentation gates — producing a
`valid_record_ids.json` mask that every training loader applies. Activations
are extracted per target model; the records are shared across target models.

Selection effects to be aware of: the target model is also the labeler; the
judge-filter mask gates the train *and* eval splits; and IFEval is a training
source, so do not evaluate these models on IFEval-derived benchmarks.

## Judge rubric provenance

The GRPO training reward used the rubric embedded in `prism/rl/judge.py`
(`SYSTEM_PROMPT`). That prompt is byte-identical to prism-eval's canonical
scoring-judge prompt, so the training reward and the published scores share
one rubric. docs/RUBRIC.md is the full scoring rubric.
