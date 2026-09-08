---
license: apache-2.0
language:
- en
library_name: peft
base_model: Qwen/Qwen3.5-9B
tags:
- peft
- lora
- interpretability
- activation-interpretability
---

# LatentQA baseline adapter for Qwen3.5-9B

This repository contains the Qwen3.5-9B LoRA adapter used for the LatentQA
comparison in the PRISM demo and evaluation. It is a baseline artifact, not a
PRISM checkpoint. The adapter reads target-model activations and answers a
free-form question about them.

The implementation is adapted from
[`aypan17/latentqa`](https://github.com/aypan17/latentqa) at revision `a2dcb6f`.
The published adapter omits duplicated embedding and language-head tensors; the
PRISM demo attaches it to a separately downloaded Qwen3.5-9B model.

## Files

| File | SHA-256 |
|---|---|
| `adapter_config.json` | `a952a74b6be55979834a491a7da2f1ddfff0f9ddc153116b692c6b6ab4a220a6` |
| `adapter_model.safetensors` | `8600f3ba51e60e53de72ff044adee962dca3c179a9b8a1212809c236a7852cd2` |

## Use

The supported integration is the comparison mode in the
[`prism`](https://github.com/Offensive-AI-Lab/prism) demo:

```bash
git clone https://github.com/Offensive-AI-Lab/prism
cd prism
uv sync --extra demo
uv run python scripts/download_baselines.py --only latentqa
uv run python demo/app.py
```

## License

The adapter files are licensed under Apache-2.0. They do not include
Qwen3.5-9B weights. The adapted LatentQA implementation is Apache-2.0 and is
attributed in the PRISM repository.

## References

- Alexander Pan, Lijie Chen, and Jacob Steinhardt. [LatentQA: Teaching LLMs to
  Decode Activations Into Natural Language](https://arxiv.org/abs/2412.08686).
- Gilad Gressel et al. [PRISM: Recovering Instruction Sets from Language Model
  Activations](https://arxiv.org/abs/2606.09563). EMNLP 2026.
