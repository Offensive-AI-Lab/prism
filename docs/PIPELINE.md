# Training pipeline

The [README](../README.md#training) gives the commands for the released Qwen
run. [Recipes](RECIPES.md) lists the settings and commands for all target models.

## 1. Prepare the data

Use the [training records](DATA_CARD.md#files-and-fields) and validity mask
for reproduction. The data card defines the fields, filtering, and splits.

### Generating new records

To build a new dataset, serve the target model through an OpenAI-compatible
endpoint. For example, in a separate vLLM environment:

```bash
vllm serve Qwen/Qwen3.5-9B --port 8089
```

From the repository root, select a separate directory for the new data:

```bash
export PRISM_DATA_DIR=/path/to/new-prism-data
export DATAGEN_BASE_URL=http://localhost:8089/v1
export DATAGEN_MODEL=Qwen/Qwen3.5-9B
scripts/generate_dataset.sh
scripts/clean_dataset.sh
```

Generation writes `prompt-only/jsonl/*.jsonl`; cleaning writes
`prompt-only/valid_record_ids.json`. The cleaned record copies under
`filtered/` are intermediate outputs, not the training input directory.
Set `PRISM_DATA_DIR` to this root when training on the new records.

The shell scripts above require exported settings. Training recipes also read
`.env`; see [`.env.example`](../.env.example) for generation overrides.

## 2. Train with SFT

Supervised fine-tuning trains a linear projection and LoRA adapters to predict
the instruction list. The projection maps frozen target-model activations into
input embeddings for the same model, used as the PRISM decoder.

### Activation extraction

Both paths read `prompt + response` and take up to the last 128 response-token
activations at the target profile's hook layer:

- **Cached (default):** the recipe extracts sharded safetensors on first use
  and reuses them for SFT and GRPO. An existing cache can be selected with
  `PRISM_PRECOMPUTED_DIR`. It must match the target model, layer, and data.
- **On-the-fly:** set `PRISM_ON_THE_FLY=1` for either recipe. Each sampled batch
  passes through the resident base model with LoRA disabled and gradients off;
  the forward stops at the hook layer. This partial forward is the additional
  model computation. GRPO records its duration as `timing/extract_s`.

Both trainers still load the target model to run the decoder. Cached extraction
trades disk space for throughput; it was used for the released checkpoints.

The recipes sort the JSONLs and use the same split function and validity mask
in both paths. The cache stores split membership at extraction time and copies
the mask beside its manifest; on-the-fly loaders find the mask next to the
JSONL directory. Keep the original records, order, and split settings to retain
the same membership.

## 3. Refine with GRPO and export

GRPO starts from the target model's SFT checkpoint. For each record, it samples
candidate instruction reports, scores them with the judge, and updates the
projection and LoRA adapters using group-relative rewards. A KL penalty limits
divergence from the frozen SFT reference.

Instruction coverage and hallucination follow the
[canonical scoring rubric](https://github.com/Offensive-AI-Lab/prism-eval/blob/main/RUBRIC.md).
All three released GRPO recipes use prioritized sampling, dynamic sampling that
rejects low-variance or near-ceiling groups, and penalties for overly long or
short reports. [Recipes](RECIPES.md) gives their settings.

The recipes select `best.pt` by validation loss for SFT and validation judge
reward for GRPO. They also save training state for resuming. Candidate scores
are written to `judge_traces.jsonl`, which
[`analyze_judge_traces.py`](../scripts/analyze_judge_traces.py) summarizes.
For custom hard-example sampling, `prism.rl.build_hard_ids` converts those traces
into IDs accepted by the trainer's `--hard-ids-json` option.

### Export and checkpoint format

Export the selected checkpoint for inference:

```bash
uv run python scripts/export_checkpoint.py \
  "$PRISM_CKPT_DIR/grpo-qwen3.5-9b-L16/best.pt"
```

This writes `exports/prism-qwen3.5-9b-grpo.pt`. The exporter infers the target
model and training method; `--out-dir` changes the output directory.

| Release field | Contents |
|---|---|
| `config` | Model and adapter configuration, with training paths and run identities removed or replaced |
| `lora_state` | LoRA adapter weights, retained in fp32 |
| `projection_state` | Projection weight and bias, converted to bf16 |
| `opt_step` | Optimizer-step metadata |

Training checkpoints additionally retain optimizer, scheduler, and validation
state; GRPO can also retain its sampling tracker and run ID. Exported checkpoints
omit that state and are for inference, not full training resumption.

These files use `torch.save` serialization. The exporter loads with
`weights_only=False`, so export only training checkpoints you trust.

### Target-model settings

[`target_models.py`](../src/prism/target_models.py) controls model loading and
tokenization. Preserve its Qwen position-ID handling, Gemma eager attention for
softcapping, and Ministral text-only LoRA scope when extending the pipeline.
