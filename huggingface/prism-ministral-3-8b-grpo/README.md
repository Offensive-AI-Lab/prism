---
license: apache-2.0
language:
- en
library_name: pytorch
base_model: mistralai/Ministral-3-8B-Instruct-2512-BF16
datasets:
- Offensive-AI-Lab/prism-training-dataset
tags:
- interpretability
- activation-interpretability
- instruction-recovery
---

# PRISM for Ministral 3 8B

This repository contains the Ministral 3 8B PRISM checkpoint from *PRISM:
Recovering Instruction Sets from Language Model Activations*. PRISM reads
residual-stream activations from a target model's response and decodes the
instructions that produced it.

The checkpoint contains a learned projection and LoRA parameters. It does not
contain Ministral weights.

## Checkpoint

| Target model | Hook layer | Activation window | Training |
|---|---:|---:|---|
| `mistralai/Ministral-3-8B-Instruct-2512-BF16` | 17 | Last 128 response tokens | SFT + GRPO |

`prism-ministral-3-8b-grpo.pt` SHA-256:
`1ad068c4928e503b9526510d1790b839d16dc0de534efa0ff148b9ee221ef128`

## Use

Use the checkpoint with
[`prism-eval`](https://github.com/Offensive-AI-Lab/prism-eval):

```bash
git clone https://github.com/Offensive-AI-Lab/prism-eval
cd prism-eval
uv sync
uv run python scripts/download_weights.py --only prism-ministral-3-8b-grpo
uv run prism-eval evaluate --config configs/main/ministral-3-8b-grpo.yaml --offline
```

The evaluation requires a separate judge endpoint; see the repository README.

## Limitations

PRISM can omit instructions or report instructions that were not present. The
released evaluation is primarily English and uses single-response examples.
Its outputs are interpretability evidence, not a safety guarantee.

## License

The checkpoint is licensed under Apache-2.0. Ministral weights are downloaded
separately and remain subject to their own license.

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
