# Training data

PRISM's training prompts come from three datasets:

| Source key | Upstream dataset | Source license | PRISM records before filtering |
|---|---|---|---:|
| `if_eval` | [google/IFEval](https://huggingface.co/datasets/google/IFEval) | Apache-2.0 | 492 |
| `if_multi_constraints` | [allenai/IF_multi_constraints_upto5](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5) | ODC-By-1.0 | 77,002 |
| `ultrachat` | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | MIT | 200,002 |

The [training dataset](https://huggingface.co/datasets/Offensive-AI-Lab/prism-training-dataset)
contains these 277,496 records and a validity mask selecting 203,589 for use.
It remains private pending redistribution review. The source licenses above
do not replace that review; the Ai2 source also lists third-party model-output
terms in its dataset card.

## Files and fields

With authorized Hugging Face access, download the records using
[`scripts/download_dataset.py`](../scripts/download_dataset.py). It places
`if_eval.jsonl`, `if_multi_constraints.jsonl`, and `ultrachat.jsonl` in
`$PRISM_DATA_DIR/prompt-only/jsonl/`, with `valid_record_ids.json` beside that
directory. The [training instructions](../README.md#1-prepare-the-data)
show the download command.

Each record has these fields:

| Field | Meaning |
|---|---|
| `id` | Stable record identifier used by the validity mask |
| `source_dataset` | Source key from the table above |
| `prompt` | Instruction-rich user request |
| `response` | Qwen3.5-9B's generated response to that request |
| `instruction_set` | Generated instruction labels, stored as a bulleted text string |
| `metadata` | Generation metadata, including `paraphrase_group_id` for related prompts |

Qwen3.5-9B generates both responses and instruction labels. Labels are generated
from the prompt alone at temperature 0.3, not inferred from the response or
written by human annotators. The fixed request for an instruction list lives in
code rather than in each record.

The same records are used for Qwen, Gemma, and Ministral training. Each target
model reads the stored `prompt + response` to produce its own activations;
the stored response is not regenerated for each target.

## How training uses the records

Both stages condition the PRISM decoder on response-token activations, not the
original prompt text.

- **SFT:** `instruction_set` is the target sequence for the cross-entropy loss.
- **GRPO:** PRISM generates candidate reports. The judge receives each report,
  `prompt`, `response`, and `instruction_set` to calculate its reward.

Activation caching changes how the inputs are stored, not which fields supply
the supervision. See the [pipeline guide](PIPELINE.md#activation-extraction).

## Filtering and splits

Rule-based checks reject empty or malformed labels, echoes of the label-generation
request, and likely truncation. An LLM judge checks whether the labels faithfully
enumerate the prompt's instructions. The final mask also rejects lists with more
than six bullets, template leakage, and labels fragmented into one- or two-word
bullets. These are label-quality filters, not content-safety filters.

The loaders split the full records before applying the mask. They use sorted
input files, seed 42, validation and test ratios of 0.1 each, and no source
stratification. Records sharing `metadata.paraphrase_group_id` stay together.
After masking, the splits contain:

| Split | Records |
|---|---:|
| Training | 162,821 |
| Validation | 20,410 |
| Test | 20,358 |

Keep the full JSONLs and mask together: deleting rejected records before
splitting would change membership. [`scripts/check_dataset.py`](../scripts/check_dataset.py)
checks record counts, IDs, and the released split membership.

## Regeneration and limitations

The [generation scripts](PIPELINE.md#generating-new-records) create a new sample;
they do not reconstruct the released records exactly. Source sampling,
paraphrase generation, and model-based filtering can change the resulting data.

The labels can contain Qwen3.5-9B's errors and omissions, and the validity mask
filters all three splits. IFEval contributes training prompts, so evaluation
on IFEval or overlapping derivatives is not an independent held-out evaluation.
