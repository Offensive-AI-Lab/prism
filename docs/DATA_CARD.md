# Training data

PRISM's training prompts come from three public datasets:

| Source key | Upstream dataset | Source license | Records before filtering |
|---|---|---|---:|
| `if_eval` | [google/IFEval](https://huggingface.co/datasets/google/IFEval) | Apache-2.0 | 492 |
| `if_multi_constraints` | [allenai/IF_multi_constraints_upto5](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5) | ODC-By-1.0 | 77,002 |
| `ultrachat` | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | MIT | 200,002 |

The [training dataset](https://huggingface.co/datasets/Offensive-AI-Lab/prism-training-dataset) contains all 277,496 records, along with a validity mask that selects the 203,589 used for training and evaluation.

## Files and fields

[`scripts/download_dataset.py`](../scripts/download_dataset.py) downloads the records, as shown in the [training instructions](../README.md#1-prepare-the-data). It puts `if_eval.jsonl`, `if_multi_constraints.jsonl`, and `ultrachat.jsonl` in `$PRISM_DATA_DIR/jsonl/`, and the mask, `valid_record_ids.json`, directly in `$PRISM_DATA_DIR`.

Each record has these fields:

| Field | Meaning |
|---|---|
| `id` | Stable record identifier, used by the validity mask |
| `source_dataset` | Source key from the table above |
| `prompt` | The user request, rich in instructions |
| `response` | Qwen3.5-9B's response to that request |
| `instruction_set` | The reference instruction list, stored as a bulleted string |
| `metadata` | Generation metadata, including a `paraphrase_group_id` that links related prompts |

Qwen3.5-9B wrote both the responses and the instruction lists. It generated each list from the prompt alone, at temperature 0.3, so the labels are neither inferred from the response nor written by people. The request used to generate them is fixed in the generation code rather than stored with each record.

Qwen, Gemma, and Ministral all train on these same records. Each target model reads the stored `prompt + response` to produce its own activations; the response isn't regenerated for each model.

## How training uses the records

In both stages, the PRISM decoder sees only response-token activations, never the prompt text.

- **SFT:** `instruction_set` is the target sequence for the cross-entropy loss.
- **GRPO:** PRISM generates candidate reports, and the judge scores each one given the `prompt`, `response`, and `instruction_set`.

Caching activations changes how the inputs are stored, not which fields supervise training. See the [pipeline guide](PIPELINE.md#activation-extraction).

## Filtering and splits

Rule-based checks reject labels that are empty or malformed, that echo the label-generation request, or that look truncated. An LLM judge then checks whether the labels faithfully list the prompt's instructions. These filters are about label quality, not content safety.

The loaders split the full set of records first and apply the mask afterward. They use sorted input files, seed 42, validation and test ratios of 0.1 each, and no stratification by source. Records that share a `metadata.paraphrase_group_id` always land in the same split. After masking, the splits contain:

| Split | Records |
|---|---:|
| Training | 162,821 |
| Validation | 20,410 |
| Test | 20,358 |

Keep the full JSONL files and the mask together. Deleting rejected records before splitting would change which records end up in each split. [`scripts/check_dataset.py`](../scripts/check_dataset.py) verifies the record counts, IDs, and released split membership.

## Regeneration and limitations

The [generation scripts](PIPELINE.md#generating-new-records) produce a new sample rather than an exact copy of the released records, since source sampling, paraphrasing, and model-based filtering can all change the result.

The labels can carry Qwen3.5-9B's errors and omissions. The validity mask applies to all three splits, so the test set went through the same filtering as the training data.

## License

This dataset combines several licenses. Each `prompt` keeps the terms of its source dataset:

| Source key | License |
|---|---|
| `if_eval` | Apache-2.0 |
| `if_multi_constraints` | ODC-By-1.0 |
| `ultrachat` | MIT |

The PRISM authors release the generated `response`, `instruction_set`, metadata, and validity mask under Apache-2.0, to the extent that they hold the rights to them. This doesn't override the source terms. In particular, Ai2's dataset card notes that some of its records contain third-party model output with separate terms. The exact upstream revisions and transformations are recorded in the dataset's `source_inventory.json`.
