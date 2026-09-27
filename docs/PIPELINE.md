# Training pipeline

This guide goes one level deeper than the [README](../README.md#training), which has the commands for the released Qwen run. The [training recipes](RECIPES.md) list the settings and commands for every target model.

## 1. Prepare the data

To reproduce the paper, use the released [training records](DATA_CARD.md#files-and-fields) and their validity mask. The data card describes the fields, filtering, and splits.

### Generating new records

To build a new dataset, first serve the target model behind an OpenAI-compatible endpoint. For example, with vLLM in its own environment:

```bash
vllm serve Qwen/Qwen3.5-9B --port 8089
```

Then, from the repository root, point generation at that server and at a fresh data directory:

```bash
export PRISM_DATA_DIR=/path/to/new-prism-data
export DATAGEN_BASE_URL=http://localhost:8089/v1
export DATAGEN_MODEL=Qwen/Qwen3.5-9B
scripts/generate_dataset.sh
scripts/clean_dataset.sh
```

Generation writes the records to `jsonl/*.jsonl`, and cleaning writes the validity mask to `valid_record_ids.json`. Cleaning also leaves filtered copies of the records under `filtered/`; these are intermediate outputs, so don't train on them. To train on the new data, set `PRISM_DATA_DIR` to this directory.

Unlike the training recipes, these two scripts don't read `.env`, so export their settings first. [`.env.example`](../.env.example) lists the optional generation and cleaning overrides.

## 2. Train with SFT

Supervised fine-tuning trains a linear projection and LoRA adapters to write the reference instruction list. The projection maps the frozen target model's activations into its input embedding space, and the same model, with the adapters enabled, acts as the PRISM decoder.

### Activation extraction

PRISM can get its activations in two ways. Both run the target model over `prompt + response` and keep up to the last 128 response-token activations at the profile's hook layer.

- **Cached (default).** On first use, the recipe extracts the activations to sharded safetensors files, and both SFT and GRPO reuse them. To reuse a cache built elsewhere, set `PRISM_PRECOMPUTED_DIR`; it must match the target model, layer, and data.
- **On the fly.** Set `PRISM_ON_THE_FLY=1` for either recipe. For each batch, the base model runs with LoRA disabled and gradients off, stopping at the hook layer. That partial forward pass is the only extra compute, and GRPO logs its duration as `timing/extract_s`.

The trainers load the target model either way, since it also serves as the decoder. Caching trades disk space for speed, and the released checkpoints were trained from a cache.

Both paths produce the same splits. The recipes sort the JSONL files and apply the same split function and validity mask in either case. The cache records split membership at extraction time and keeps a copy of the mask beside its manifest, while on-the-fly loaders look for the mask next to the JSONL directory. As long as you keep the original records, their order, and the split settings, membership stays the same.

## 3. Refine with GRPO

GRPO starts from the SFT checkpoint for the same target model. For each record it samples several candidate reports, has the judge score them, and updates the projection and LoRA adapters based on how each candidate compares with the rest of its group. A KL penalty keeps the model close to the frozen SFT reference.

The judge scores instruction coverage and hallucination following the [canonical scoring rubric](https://github.com/Offensive-AI-Lab/prism-eval/blob/main/RUBRIC.md). All three released GRPO recipes use prioritized sampling, dynamic sampling that skips groups whose rewards barely vary or are already near the ceiling, and penalties for reports that are too long or too short. The [training recipes](RECIPES.md#grpo-settings) give the exact values.

SFT keeps the checkpoint with the lowest validation loss as `best.pt`, and GRPO keeps the one with the highest validation judge reward. Both also save training state so that runs can resume. GRPO writes every candidate's scores to `judge_traces.jsonl`, which [`analyze_judge_traces.py`](../scripts/analyze_judge_traces.py) summarizes. For custom hard-example sampling, `prism.rl.build_hard_ids` turns those traces into IDs for the trainer's `--hard-ids-json` option.

## 4. Export

Export the selected checkpoint for inference:

```bash
uv run python scripts/export_checkpoint.py "$PRISM_CKPT_DIR/grpo-qwen3.5-9b-L16/best.pt"
```

This writes `exports/prism-qwen3.5-9b-grpo.pt`. The exporter works out the target model and training method on its own; pass `--out-dir` to write somewhere else.

### Checkpoint format

| Field | Contents |
|---|---|
| `config` | Model and adapter configuration, with training paths and run identifiers removed or replaced |
| `lora_state` | LoRA adapter weights, kept in fp32 |
| `projection_state` | Projection weight and bias, converted to bf16 |
| `opt_step` | The optimizer step at which the checkpoint was saved |

Training checkpoints also hold optimizer, scheduler, and validation state, and GRPO checkpoints can include the sampling tracker and run ID. Exported checkpoints drop all of this, so they're meant for inference and can't fully resume training.

Checkpoints are saved with `torch.save`, and the exporter loads them with `weights_only=False`, so only export checkpoints you trust.

## Target-model settings

[`target_models.py`](../src/prism/target_models.py) handles model loading and tokenization. If you extend the pipeline to new models, keep its Qwen position-ID handling, the eager attention Gemma needs for softcapping, and Ministral's text-only LoRA scope.
