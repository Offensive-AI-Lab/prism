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

# Activation Oracles baseline adapter for Qwen3.5-9B

This repository contains the Qwen3.5-9B LoRA adapter used for the Activation
Oracles comparison in the PRISM demo and evaluation. It is a baseline artifact,
not a PRISM checkpoint. The adapter verbalizes information from target-model
activations in response to a free-form question.

The implementation is adapted from
[`adamkarvonen/activation_oracles`](https://github.com/adamkarvonen/activation_oracles)
at revision `55f153f`.

## Files

| File | SHA-256 |
|---|---|
| `adapter_config.json` | `b049c48d32dfbb25e605949b6859dbe2eb53bd680d36eb6171f82e97003306cc` |
| `adapter_model.safetensors` | `1830598a70e652d4bf5a39de4439d7683e8e5825cc5c053b66cdcbbe187e1612` |

## Use

The supported integration is the comparison mode in the
[`prism`](https://github.com/Offensive-AI-Lab/prism) demo:

```bash
git clone https://github.com/Offensive-AI-Lab/prism
cd prism
uv sync --extra demo
uv run python scripts/download_baselines.py --only ao
uv run python demo/app.py
```

## License

The adapter files are licensed under Apache-2.0. They do not include
Qwen3.5-9B weights. The adapted Activation Oracles implementation is MIT,
Copyright (c) 2025 Adam Karvonen, and is attributed in the PRISM repository.

## References

- Adam Karvonen et al. [Activation Oracles: Training and Evaluating LLMs as
  General-Purpose Activation Explainers](https://arxiv.org/abs/2512.15674).
- Gilad Gressel et al. [PRISM: Recovering Instruction Sets from Language Model
  Activations](https://arxiv.org/abs/2606.09563). EMNLP 2026.
