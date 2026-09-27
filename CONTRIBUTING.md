# Contributing to PRISM

## Development setup

```bash
uv sync --extra dev
uv run pytest tests -q
```

The default tests should run on CPU, with no model downloads, credentials, or live judge endpoint.

## Submitting a change

- Keep each change focused on one thing.
- Add tests for changed logic, and update the docs when behavior changes.
- Describe any effect on data selection, training, or compatibility with existing checkpoints and activation caches.
- Keep generated data, activations, checkpoints, credentials, and machine-specific paths out of commits.

## Target models

Model-specific loader, attention, tokenizer, and LoRA changes belong in `src/prism/target_models.py`. Add tests for any assumption that affects where the response starts and ends, and document model-specific memory or kernel requirements.

## Data and scoring

Changing source data, generation prompts, filters, or the validity mask changes the training dataset, so update the [data card](docs/DATA_CARD.md) to match.

Judge prompts and rewards affect both training and evaluation. Keep the training judge aligned with the [canonical scoring rubric](https://github.com/Offensive-AI-Lab/prism-eval/blob/main/RUBRIC.md), and record recipe changes in the [training recipes](docs/RECIPES.md). Calibration artifacts belong in prism-eval.
