---
license: apache-2.0
language:
- en
library_name: pytorch
base_model: Qwen/Qwen3.5-9B
datasets:
- Offensive-AI-Lab/prism-training-dataset
tags:
- interpretability
- activation-interpretability
- instruction-recovery
---

# PRISM without reinforcement learning for Qwen3.5-9B

This repository contains the supervised-only PRISM checkpoint used for the
"PRISM w/o RL" result in *PRISM: Recovering Instruction Sets from Language
Model Activations*. It reads residual-stream activations from a target model's
response and decodes the instructions that produced it.

The checkpoint contains a learned projection and LoRA parameters. It does not
contain Qwen3.5-9B weights.

## Checkpoint

| Target model | Hook layer | Activation window | Training |
|---|---:|---:|---|
| `Qwen/Qwen3.5-9B` | 16 | Last 128 response tokens | SFT |

`prism-qwen3.5-9b-sft.pt` SHA-256:
`347cc6c6674a839dd995633a057f9ddb6cb45e153f912671445eed66c6a56002`

On the paper's 1,000-record evaluation suite, this checkpoint obtains 0.653
mean reward, 0.691 mean coverage, and 0.037 mean hallucination rate. These are
the results reported in the paper.

## Use

Use the checkpoint with
[`prism-eval`](https://github.com/Offensive-AI-Lab/prism-eval):

```bash
git clone https://github.com/Offensive-AI-Lab/prism-eval
cd prism-eval
uv sync
uv run python scripts/download_weights.py --only prism-qwen3.5-9b-sft
uv run prism-eval evaluate --config configs/main/qwen3.5-9b-sft.yaml --offline
```

The evaluation requires a separate judge endpoint; see the repository README.
For the interactive demo and training code, use
[`prism`](https://github.com/Offensive-AI-Lab/prism).

## Limitations

PRISM can omit instructions or report instructions that were not present. The
released evaluation is primarily English and uses single-response examples.
Its outputs are interpretability evidence, not a safety guarantee.

## License

The checkpoint is licensed under Apache-2.0. Qwen3.5-9B is downloaded
separately and remains subject to its own license.

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
