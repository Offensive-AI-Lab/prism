# PRISM

Training code for [PRISM: Recovering Instruction Sets from Language Model
Activations](https://arxiv.org/abs/2606.09563), accepted to the EMNLP 2026 Main
Conference.

PRISM decodes the instructions represented in a language model's hidden states.
Given residual-stream activations from a frozen target model, a learned
projection and LoRA-adapted decoder produce an instruction report. Training has
two stages: supervised fine-tuning (SFT) on generated oracle reports, followed
by judge-guided group relative policy optimization (GRPO).

This repository contains the data-generation and training pipeline. Evaluation
code, benchmark data, released checkpoints, and published results are in
[`Offensive-AI-Lab/prism-eval`](https://github.com/Offensive-AI-Lab/prism-eval).

## Repository scope

| Component | Location |
|---|---|
| Oracle data generation and filtering | `src/prism/datagen/` |
| Activation extraction and sharding | `src/prism/activations/` |
| SFT | `src/prism/sft/` |
| GRPO | `src/prism/rl/` |
| Reproduction recipes | `recipes/` |

The released training dataset is on Hugging Face (`Offensive-AI-Lab/prism-training-dataset`); `uv run python scripts/download_dataset.py` fetches and verifies it, and the generation scripts can build a fresh dataset instead.

## Requirements

- Linux or another environment capable of running the Bash recipes
- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- A CUDA GPU for activation extraction and training
- Access to the target model weights on Hugging Face
- An OpenAI-compatible judge endpoint for data filtering and GRPO

The released runs used a single GPU with approximately 95 GB of memory. The
judge was served separately with vLLM. Hardware and runtime notes are collected
in [the pipeline guide](docs/PIPELINE.md).

## Installation

```bash
git clone https://github.com/Offensive-AI-Lab/prism.git
cd prism
uv sync
cp .env.example .env
```

Set the storage roots and credentials in `.env`:

```dotenv
PRISM_DATA_DIR=/path/to/prism-data
PRISM_CKPT_DIR=/path/to/prism-checkpoints
HF_TOKEN=
PRISM_JUDGE_BASE_URL=http://localhost:8088/v1
PRISM_JUDGE_MODEL=gemma4-31B-it
PRISM_JUDGE_API_KEY=not-needed
```

Values already present in the environment take precedence over `.env`.

Optional dependency groups are `datagen` (local vLLM generation),
`bertscore`, and `dev`. Install one with, for example,
`uv sync --extra datagen`.

## Training pipeline

The released checkpoints were produced by the following sequence:

1. Generate oracle examples with a target model.
2. Filter the generated labels and write a valid-record mask.
3. Extract and shard response-token activations.
4. Train the projection and LoRA parameters with SFT.
5. Continue training with judge-guided GRPO.
6. Export an inference-only checkpoint for `prism-eval`.

Activation extraction is normally invoked by the SFT and GRPO recipes. The
cache is reused when the target-model profile, hook layer, and output directory
match.

Both trainers can also extract activations on the fly from the oracle JSONL
files (`PRISM_ON_THE_FLY=1` before a recipe, or `--dataset-paths` on the
module) using the target model that is already resident for training. The main
additional model-compute cost is one no-grad forward up to the hook layer per
batch. The precomputed cache remains the higher-throughput path and was used
for all reported results and released checkpoints. See
[docs/PIPELINE.md](docs/PIPELINE.md) §2.

### 1. Generate and filter oracle data

Serve the target model at the endpoint configured by `DATAGEN_BASE_URL` and
`DATAGEN_MODEL`, then run:

```bash
scripts/generate_dataset.sh
scripts/clean_dataset.sh
```

The expected output is
`$PRISM_DATA_DIR/prompt-only/jsonl/*.jsonl` plus
`$PRISM_DATA_DIR/prompt-only/valid_record_ids.json`.

### 2. Run SFT

```bash
recipes/sft_qwen3.5-9b.sh
```

Equivalent recipes are provided for Gemma 2 9B and Ministral 3 8B. Each recipe
loads the matching target-model profile from `src/prism/target_models.py`,
builds or reuses the activation cache, and writes checkpoints below
`$PRISM_CKPT_DIR`.

### 3. Run GRPO

Start the judge server in a separate process and then run the matching GRPO
recipe:

```bash
scripts/serve_judge.sh
recipes/grpo_qwen3.5-9b.sh
```

GRPO starts from the SFT checkpoint for the same target model. The exact
released hyperparameters are recorded in [docs/RECIPES.md](docs/RECIPES.md);
the Python module defaults are intended for development and are not the
reproduction configuration.

### 4. Export a checkpoint

```bash
uv run python scripts/export_checkpoint.py \
  "$PRISM_CKPT_DIR/<run>/best.pt"
```

The export removes optimizer and scheduler state and writes the format consumed
by `prism-eval`. The schema and invariants are documented in
[docs/CHECKPOINT_FORMAT.md](docs/CHECKPOINT_FORMAT.md).

### Smoke runs

Set `SMOKE=1` before an SFT or GRPO recipe to use a small data slice and separate
output directories:

```bash
SMOKE=1 recipes/sft_qwen3.5-9b.sh
SMOKE=1 recipes/grpo_qwen3.5-9b.sh
```

Smoke runs check configuration, model loading, activation extraction, and
checkpoint writing. They are not suitable for comparing metrics.

## Released checkpoints

| Checkpoint | Target model | Hook layer | Training |
|---|---|---:|---|
| `prism-qwen3.5-9b-sft.pt` | `Qwen/Qwen3.5-9B` | 16 | SFT |
| `prism-qwen3.5-9b-grpo.pt` | `Qwen/Qwen3.5-9B` | 16 | SFT + GRPO |
| `prism-gemma-2-9b-it-grpo.pt` | `google/gemma-2-9b-it` | 21 | SFT + GRPO |
| `prism-ministral-3-8b-grpo.pt` | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | 17 | SFT + GRPO |

Download and evaluate these files through
[`prism-eval`](https://github.com/Offensive-AI-Lab/prism-eval). The Qwen GRPO
checkpoint is the primary model reported in the paper.

## Adding a target model

Target-specific behavior is defined in `src/prism/target_models.py`. A new
profile specifies the Hugging Face model ID, hook layer, model loader, attention
implementation, LoRA scope, and chat-template adjustments. Before training:

```bash
uv run python scripts/check_chat_template.py <profile-name>
```

Then add SFT and GRPO recipes based on the closest existing model and update the
exporter's model-tag mapping. If extraction requires a different batch size or
attention implementation, update the corresponding profile and recipe helper.

## Reproducibility notes

- Regenerating the oracle dataset does not reproduce the original JSONL files
  bit for bit. Model sampling, serving order, and judge filtering can change the
  retained examples.
- The recipes, rather than the dataclass defaults in the training modules,
  define the released runs.
- The released runs trained from the precomputed activation cache;
  [docs/PIPELINE.md](docs/PIPELINE.md) describes how on-the-fly extraction
  relates to it.
- Do not use the released monitors for evaluation on IFEval-derived benchmarks;
  IFEval is one of the oracle-data sources.

## Documentation

| Document | Contents |
|---|---|
| [Pipeline](docs/PIPELINE.md) | End-to-end data, activation, SFT, GRPO, and export flow |
| [Released recipes](docs/RECIPES.md) | Hyperparameters and checkpoint provenance |
| [Data card](docs/DATA_CARD.md) | Source datasets, generated labels, and licensing |
| [Checkpoint format](docs/CHECKPOINT_FORMAT.md) | Training and release checkpoint schemas |
| [Scoring rubric](docs/RUBRIC.md) | Coverage and hallucination rubric |
| [Ablations](docs/ABLATIONS.md) | Training-side ablation commands |

## Citation

```bibtex
@inproceedings{gressel2026prism,
  title     = {PRISM: Recovering Instruction Sets from Language Model Activations},
  author    = {Gressel, Gilad and Pankajakshan, Rahul and Diament, Julia and
               Hudis, Efim and Achuthan, Krishnashree and Mirsky, Yisroel},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in
               Natural Language Processing},
  year      = {2026},
  url       = {https://arxiv.org/abs/2606.09563}
}
```

## License

The code is licensed under the [Apache License 2.0](LICENSE). Generated data may
also be subject to the licenses and terms of the source datasets and target
models; see [docs/DATA_CARD.md](docs/DATA_CARD.md).
