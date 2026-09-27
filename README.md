# PRISM

[![arXiv](https://img.shields.io/badge/arXiv-2606.09563-b31b1b.svg)](https://arxiv.org/abs/2606.09563)
[![Hugging Face checkpoints](https://img.shields.io/badge/Hugging_Face-checkpoints-FFD21E.svg)](https://huggingface.co/Offensive-AI-Lab/models)
[![Hugging Face dataset](https://img.shields.io/badge/Hugging_Face-dataset-FFD21E.svg)](https://huggingface.co/datasets/Offensive-AI-Lab/prism-training-dataset)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

PRISM reads a language model's activations as it responds and writes out the instructions represented in them, from the user's actual request to hidden objectives and prompt injections the user never sees.

![A hidden instruction in a file tells Qwen3.5-9B to skip an approval step. The response looks normal, but PRISM recovers the injected instruction from the activations.](docs/why-prism.svg)

This is the official code for [PRISM: Recovering Instruction Sets from Language Model Activations](https://arxiv.org/abs/2606.09563), accepted to the EMNLP 2026 Main Conference. It includes an interactive demo and the full training pipeline. The evaluation code, benchmarks, and paper results live in [prism-eval](https://github.com/Offensive-AI-Lab/prism-eval).

## Try PRISM

You'll need Python 3.13+, [uv](https://docs.astral.sh/uv/), and a CUDA GPU with 24 GB of memory.

```bash
git clone https://github.com/Offensive-AI-Lab/prism.git
cd prism
uv sync --extra demo
uv run python demo/app.py
```

Open [localhost:7860](http://127.0.0.1:7860), pick an example or type your own prompt, and click **Generate response**. Then click **Recover instructions** to see what PRISM reads from the model's activations.

![The PRISM demo: the target model's response, the response tokens PRISM reads, and the recovered instructions](docs/demo.png)

The first launch downloads Qwen3.5-9B (about 18 GB) and the PRISM checkpoints (about 266 MB each). The **Variant** menu switches to PRISM without RL, and **Compare baselines** runs LatentQA and Activation Oracles on the same response, which downloads about 580 MB more the first time.

## What's in this repo

| Path | Contents |
|---|---|
| [`demo/`](demo/) | The local web demo |
| [`src/prism/`](src/prism/) | Activation extraction, the SFT and GRPO trainers, and dataset generation |
| [`recipes/`](recipes/) | One script per training run in the paper |
| [`scripts/`](scripts/) | Helpers to download the dataset, serve the judge, and export checkpoints |
| [`docs/`](docs/) | The [pipeline guide](docs/PIPELINE.md), [training recipes](docs/RECIPES.md), [ablations](docs/ABLATIONS.md), and [data card](docs/DATA_CARD.md) |

## Checkpoints

A PRISM checkpoint is only a learned projection and a set of LoRA adapters, so it runs on top of the original target model rather than replacing it. The demo uses the two Qwen checkpoints, and prism-eval can run all four.

| Checkpoint | Target model | Layer | Training |
|---|---|---:|---|
| [PRISM — Qwen](https://huggingface.co/Offensive-AI-Lab/prism-qwen3.5-9b-grpo) | `Qwen/Qwen3.5-9B` | 16 | SFT + GRPO (main model in the paper) |
| [PRISM w/o RL — Qwen](https://huggingface.co/Offensive-AI-Lab/prism-qwen3.5-9b-sft) | `Qwen/Qwen3.5-9B` | 16 | SFT only |
| [PRISM — Gemma](https://huggingface.co/Offensive-AI-Lab/prism-gemma-2-9b-it-grpo) | `google/gemma-2-9b-it` | 21 | SFT + GRPO |
| [PRISM — Ministral](https://huggingface.co/Offensive-AI-Lab/prism-ministral-3-8b-grpo) | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | 17 | SFT + GRPO |

## How it works

PRISM takes the residual-stream activations of up to the last 128 response tokens at one layer of the target model. A learned projection maps those activations into the model's input embedding space, and the same model, with LoRA adapters switched on, decodes them into a list of instructions. The target model's own weights stay frozen; training only updates the projection and the adapters.

![PRISM architecture: activation extraction, instruction recovery, and GRPO training](docs/prism-architecture.png)

Training has two stages. Supervised fine-tuning (SFT) first teaches PRISM to write instruction lists. GRPO then refines them with an LLM judge, which scores each candidate list against the reference instructions, rewarding the instructions it recovers and penalizing the ones it makes up.

## Training

The released models were trained on Linux with a 95 GB GPU and a separate vLLM server for the judge. To set up, install the dependencies, copy the example config, and export where data and checkpoints should go:

```bash
uv sync
cp .env.example .env
export PRISM_DATA_DIR=/path/to/prism-data
export PRISM_CKPT_DIR=/path/to/prism-checkpoints
```

Everything else goes in `.env`: the judge endpoint for GRPO, a Hugging Face token if you train on the gated Gemma model, and a W&B key if you want experiment tracking.

### 1. Prepare the data

The [training dataset](https://huggingface.co/datasets/Offensive-AI-Lab/prism-training-dataset) pairs prompts from IFEval, IF Multi-Constraints, and UltraChat with model responses and reference instruction lists. Download it into `$PRISM_DATA_DIR`:

```bash
uv run python scripts/download_dataset.py
```

The [data card](docs/DATA_CARD.md) covers sources, fields, and filtering. To build your own dataset instead, see the [pipeline guide](docs/PIPELINE.md#generating-new-records).

### 2. Train with SFT

```bash
recipes/sft_qwen3.5-9b.sh
```

The first run extracts response-token activations and caches them to disk, and both training stages reuse that cache. If disk space is tight, set `PRISM_ON_THE_FLY=1` to extract activations during training instead. It's slower, but needs no cache; the [pipeline guide](docs/PIPELINE.md#activation-extraction) explains the trade-off.

### 3. Refine with GRPO

GRPO needs an LLM judge behind an OpenAI-compatible endpoint, which you set in `.env`:

```dotenv
PRISM_JUDGE_BASE_URL=http://localhost:8088/v1
PRISM_JUDGE_MODEL=gemma4-31B-it
PRISM_JUDGE_API_KEY=not-needed
```

We used `google/gemma-4-31B-it` with reasoning turned off. To host it yourself, run `scripts/serve_judge.sh` on a GPU machine and point `PRISM_JUDGE_BASE_URL` at it. Then start GRPO, which picks up the best SFT checkpoint automatically:

```bash
recipes/grpo_qwen3.5-9b.sh
```

### 4. Export

Exporting drops the optimizer and other training state, leaving a small checkpoint that prism-eval can load. This writes `exports/prism-qwen3.5-9b-grpo.pt`:

```bash
uv run python scripts/export_checkpoint.py "$PRISM_CKPT_DIR/grpo-qwen3.5-9b-L16/best.pt"
```

The [training recipes](docs/RECIPES.md) list every setting and include the Gemma and Ministral runs. The [ablations](docs/ABLATIONS.md) page covers the layer sweep and the seed runs.

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

The code and PRISM checkpoints are released under the [Apache 2.0 license](LICENSE). Target models aren't included and keep their own licenses. The training data draws on sources with several licenses, listed in the [data card](docs/DATA_CARD.md), and the vendored baseline code in [`demo/third_party/`](demo/third_party/) keeps its original licenses.
