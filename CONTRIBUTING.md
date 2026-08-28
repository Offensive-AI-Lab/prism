# Contributing to PRISM

This repository contains the training pipeline used to produce the released
PRISM checkpoints. Changes that alter data selection, activation extraction,
losses, rewards, or checkpoint selection can affect reproducibility and should
be documented explicitly.

## Development setup

```bash
uv sync --extra dev
uv run pytest tests -q
```

Tests are CPU-only unless a change specifically introduces a GPU integration
test. Do not add tests that require private model weights, API credentials, or a
live judge endpoint.

## Submitting a change

1. Keep the change limited to one concern.
2. Update the relevant recipe or document when behavior changes.
3. Add or update tests for deterministic logic.
4. Record any compatibility impact on released checkpoints or activation
   caches.

Avoid committing generated datasets, activation shards, model checkpoints,
tokens, or local storage paths.

## Adding a target model

Add a profile to `src/prism/target_models.py`, verify its chat template with
`scripts/check_chat_template.py`, and create matching SFT and GRPO recipes. The
profile must specify the model loader, hook layer, attention implementation,
LoRA scope, and any chat-template adaptations. Also update the model-tag mapping
in `scripts/export_checkpoint.py`.

Document model-specific memory requirements or kernel dependencies in the
profile and in `docs/KNOWN_ISSUES.md` when they affect reproducibility.

## Changing the training data

The generated data are not versioned in this repository. A change to source
datasets, generation prompts, filters, judge prompts, or the valid-record mask
defines a new data version. Describe the change in `docs/DATA_CARD.md` and do
not present the resulting checkpoint as a reproduction of the released run.

## Changing the judge or reward

Judge prompts and reward semantics are part of the training method. Changes
require an updated rubric, calibration against human labels where applicable,
and a note in `docs/RECIPES.md`. Keep compatibility behavior needed to inspect
or export existing checkpoints.
