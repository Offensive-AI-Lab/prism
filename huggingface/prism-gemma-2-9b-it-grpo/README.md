---
license: apache-2.0
language:
- en
library_name: pytorch
base_model: google/gemma-2-9b-it
datasets:
- Offensive-AI-Lab/prism-training-dataset
tags:
- interpretability
- activation-interpretability
- instruction-recovery
---

# PRISM for Gemma 2 9B IT

This repository contains the Gemma 2 9B IT PRISM checkpoint from *PRISM:
Recovering Instruction Sets from Language Model Activations*. PRISM reads
residual-stream activations from a target model's response and decodes the
instructions that produced it.

The checkpoint contains a learned projection and LoRA parameters. It does not
contain Gemma weights.

## Checkpoint

| Target model | Hook layer | Activation window | Training |
|---|---:|---:|---|
| `google/gemma-2-9b-it` | 21 | Last 128 response tokens | SFT + GRPO |

`prism-gemma-2-9b-it-grpo.pt` SHA-256:
`45e475030a31286e4b25a3404f04bc17b7624156fa7852a6362fbdf59ce09a4a`

## Use

Use the checkpoint with
[`prism-eval`](https://github.com/Offensive-AI-Lab/prism-eval):

```bash
git clone https://github.com/Offensive-AI-Lab/prism-eval
cd prism-eval
uv sync
uv run python scripts/download_weights.py --only prism-gemma-2-9b-it-grpo
uv run prism-eval evaluate --config configs/main/gemma-2-9b-it-grpo.yaml --offline
```

The evaluation requires access to the gated target model and a separate judge
endpoint; see the repository README.

## Limitations

PRISM can omit instructions or report instructions that were not present. The
released evaluation is primarily English and uses single-response examples.
Its outputs are interpretability evidence, not a safety guarantee.

## License

The PRISM checkpoint is licensed under Apache-2.0. Gemma weights are downloaded
separately and remain subject to the Gemma terms of use.

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
