# PRISM

PRISM reads a language model's activations and produces a list of the
instructions represented in them. The reports can reveal ordinary requests,
behavioral constraints, hidden objectives, and injected instructions.

Code for [PRISM: Recovering Instruction Sets from Language Model
Activations](https://arxiv.org/abs/2606.09563), accepted to the EMNLP 2026
Main Conference. This repository includes the local demo and training pipeline;
[prism-eval](https://github.com/Offensive-AI-Lab/prism-eval) contains the
evaluation code, benchmark, and results.

## Why PRISM?

Models can act on instructions that are not stated in their responses. PRISM
decodes those instructions from response-token activations, including ordinary
constraints, hidden objectives, and injected instructions.

![Ordinary, hidden, and injected instructions shape a model response; PRISM recovers them from response-token activations](docs/why-prism.svg)

## Try PRISM

The demo runs on your own GPU. Use Python 3.13 or later and
[uv](https://docs.astral.sh/uv/). We recommend a CUDA GPU with 24 GB of memory
available. No judge endpoint is needed.

```bash
git clone https://github.com/Offensive-AI-Lab/prism.git
cd prism
uv sync --extra demo
uv run python demo/app.py
```

Open [localhost:7860](http://127.0.0.1:7860), choose an example or enter your own
prompt, and generate a response. Then run PRISM to recover the instructions
from that response's activations.

![PRISM local demo](docs/demo.png)

The three panels show the target-model response, the response tokens whose
activations PRISM reads, and the recovered instruction set.

On first launch, the demo downloads Qwen3.5-9B (about 18 GB) and the two Qwen
PRISM checkpoints (about 266 MB each). The target model uses the Hugging Face
cache; PRISM checkpoints go in `./checkpoints`. Change the latter with
`--checkpoint-dir`.

The default mode uses the final PRISM checkpoint. You can also select
**PRISM w/o RL** or compare with LatentQA and Activation Oracles. The first
comparison downloads about 580 MB of additional adapters.

## Checkpoints

Each checkpoint contains a learned projection and LoRA adapters, not the target
model's weights.

| Checkpoint | Target model | Hook layer | Training |
|---|---|---:|---|
| [PRISM — Qwen](https://huggingface.co/Offensive-AI-Lab/prism-qwen3.5-9b-grpo) | `Qwen/Qwen3.5-9B` | 16 | SFT + GRPO; main paper result |
| [PRISM w/o RL — Qwen](https://huggingface.co/Offensive-AI-Lab/prism-qwen3.5-9b-sft) | `Qwen/Qwen3.5-9B` | 16 | SFT only |
| [PRISM — Gemma](https://huggingface.co/Offensive-AI-Lab/prism-gemma-2-9b-it-grpo) | `google/gemma-2-9b-it` | 21 | SFT + GRPO |
| [PRISM — Ministral](https://huggingface.co/Offensive-AI-Lab/prism-ministral-3-8b-grpo) | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | 17 | SFT + GRPO |

## How it works

PRISM takes residual-stream activations from up to the last 128 response tokens at a
selected layer of the frozen target model. A learned projection maps those
activations into input embeddings. The same model, with LoRA adapters enabled,
decodes them into an instruction report.

![PRISM architecture: activation extraction, instruction recovery, and GRPO training](docs/prism-architecture.png)

The target model's base weights remain frozen. Training updates only the
projection and LoRA adapters; GRPO scores candidate instruction reports against
the reference instruction set. The [source PDF](docs/prism-architecture.pdf)
is included for print use.

Training has three stages:

1. Prepare prompts, target-model responses, and instruction-set labels.
2. Train the projection and LoRA adapters with supervised fine-tuning (SFT).
3. Refine the reports with judge-guided group relative policy optimization (GRPO).

## Training

The Bash recipes require Linux or a compatible environment. The released
training runs used a GPU with approximately 95 GB of memory and a separate
vLLM server for the judge. These are training requirements, not demo requirements.

Install the training environment and choose storage directories:

```bash
uv sync
cp .env.example .env
export PRISM_DATA_DIR=/path/to/prism-data
export PRISM_CKPT_DIR=/path/to/prism-checkpoints
```

The recipes also read these settings from `.env`. Exporting them makes them
available to the standalone download and export commands below.

### 1. Prepare the data

The [training dataset](https://huggingface.co/datasets/Offensive-AI-Lab/prism-training-dataset)
contains prompts from IFEval, IF Multi-Constraints, and UltraChat, paired with
target-model responses and generated instruction lists.

```bash
uv run python scripts/download_dataset.py
uv run python scripts/check_dataset.py --dataset-dir "$PRISM_DATA_DIR/prompt-only"
```

To generate and filter a new dataset instead, see the
[pipeline guide](docs/PIPELINE.md). Sources, fields, and filtering are described
in the [data card](docs/DATA_CARD.md).

### 2. Train with SFT

```bash
recipes/sft_qwen3.5-9b.sh
```

The recipe extracts and caches response-token activations on first use.
Both SFT and GRPO also support on-the-fly extraction: set `PRISM_ON_THE_FLY=1`
before running a recipe. This avoids the cache but adds a no-gradient
activation-extraction forward per batch. The released runs used the cache;
see the [pipeline guide](docs/PIPELINE.md#activation-extraction)
for the differences between the two paths.

### 3. Refine with GRPO and export

Configure an OpenAI-compatible judge endpoint in `.env`:

```dotenv
PRISM_JUDGE_BASE_URL=http://localhost:8088/v1
PRISM_JUDGE_MODEL=gemma4-31B-it
PRISM_JUDGE_API_KEY=not-needed
```

The released runs used `google/gemma-4-31B-it` with reasoning disabled.
To serve it yourself, run `scripts/serve_judge.sh` in a separate terminal on
the judge's GPU host, then use that host's address above.

Run GRPO from the SFT checkpoint and export the resulting model:

```bash
recipes/grpo_qwen3.5-9b.sh
uv run python scripts/export_checkpoint.py \
  "$PRISM_CKPT_DIR/grpo-qwen3.5-9b-L16/best.pt"
```

The export removes training state and produces a checkpoint for `prism-eval`.
[Released recipes](docs/RECIPES.md) lists the training settings and corresponding
Gemma and Ministral commands.

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

Project code and PRISM-authored checkpoint files are licensed under the
[Apache License 2.0](LICENSE). Target-model weights are not included and remain
under their respective licenses. The training dataset contains material under
multiple source licenses; see the [data card](docs/DATA_CARD.md). Vendored
baseline code retains the licenses in [demo/third_party/](demo/third_party/).
