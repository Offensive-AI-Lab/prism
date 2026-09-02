# Data card

The released training dataset lives at
`Offensive-AI-Lab/prism-training-dataset` on Hugging Face
(`scripts/download_dataset.py` fetches and verifies it;
`scripts/check_dataset.py` revalidates records, counts and split membership).
This card documents what it contains, from what sources, and under what
licenses.

## Oracle training data

| field | contents |
|---|---|
| `prompt` | instruction-rich user prompt, drawn from the sources below |
| `response` | target model's answer to `prompt` (sampled) |
| `retrieval_prompt` | fixed oracle prompt ("summarize the instructions you were given") |
| `instruction_set` | oracle label: bullet list of the instructions in `prompt` (prompt-only: generated from the prompt alone, temp 0.3) |
| `metadata` | source, paraphrase group id, generation parameters |

### Sources pulled at generation time

| source key | upstream | license notes |
|---|---|---|
| `if_eval` | google/IFEval (HF) | Apache-2.0 |
| `if_multi_constraints` | IF-multi-constraints-style prompts | check the configured HF dataset's card |
| `ultrachat` | HuggingFaceH4/ultrachat_200k (HF) | MIT per its HF dataset card (a filtered derivative of the UltraChat corpus) — verify the card before redistributing derived data |

Generated `response`/`instruction_set` are outputs of the target model
(Qwen3.5-9B for the released runs) — additionally subject to the target
model's license/usage terms. **If you redistribute generated JSONLs, you
inherit all of the above**; that is why this repo ships scripts only.

### Labels

`instruction_set` is the **prompt-only** oracle label: the target model is shown
`prompt` alone and asked to list the instructions it contains (temperature
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
