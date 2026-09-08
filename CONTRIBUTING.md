# Contributing to PRISM

## Development setup

```bash
uv sync --extra dev
uv run pytest tests -q
```

Keep default tests CPU-only and independent of model downloads, credentials,
or a live judge endpoint.

## Submitting a change

- Keep each change limited to one concern.
- Add tests for changed logic and update instructions when behavior changes.
- Describe effects on data selection, training, or compatibility with existing
  checkpoints and activation caches.
- Keep generated data, activations, checkpoints, credentials, and machine-specific
  paths out of commits.

## Target models

Keep loader, attention, tokenizer, and LoRA adaptations in
`src/prism/target_models.py`. Add tests for assumptions that affect response
boundaries, and document model-specific memory or kernel requirements.

## Data and scoring

Changes to source data, generation prompts, filters, or the validity mask change
the training dataset. Update the [data card](docs/DATA_CARD.md) accordingly.

Changes to judge prompts or rewards affect both training and evaluation. Keep
the training judge aligned with the
[canonical scoring rubric](https://github.com/Offensive-AI-Lab/prism-eval/blob/main/RUBRIC.md)
and record recipe changes in [Training recipes](docs/RECIPES.md). Calibration
artifacts belong in `prism-eval`.
