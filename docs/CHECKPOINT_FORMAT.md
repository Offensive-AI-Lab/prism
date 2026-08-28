# Checkpoint formats

## Training checkpoints (written by the trainers)

`prism.sft.train` saves `{step, best}.pt` dicts with:

```
config            full training config dict
lora_state        PEFT adapter state dict (fp32)
projection_state  {"proj.weight" [D,D], "proj.bias" [D]}
opt_step          optimizer step
optimizer_state / scheduler_state
best_val_loss

(Checkpoints written by earlier versions of the trainer additionally carry
`encoder_state: None` and encoder-ablation config keys — both ignored by
this repo's loaders and by prism-eval.)
```

`prism.rl.train` saves the same core plus RL bookkeeping:

```
val_reward / best_reward
wandb_run_id                (dropped when resuming into a fresh run)
reward_tracker_state        (prioritized-sampling runs only)
```

## Release format (what prism-eval loads)

`scripts/export_checkpoint.py` converts a training checkpoint into the format
consumed by [prism-eval](https://github.com/Offensive-AI-Lab/prism-eval)'s
runner (`prism_eval/runners/prism.py`):

```
config            sanitized training config (no local paths / W&B identity)
lora_state        unchanged (fp32)
projection_state  cast to bf16 (the runner loads it to bf16 anyway)
opt_step          unchanged
```

Everything else is stripped. The runner reads from `config`: `model_id`,
`hook_layer`, `lora_r`, `lora_alpha`, `lora_target_modules`,
`max_act_tokens`, `skip_prompt_b`, `projection_dim`, `_use_projection`
(plus optional generation/prompt fields it defaults when absent).

## Invariants of the released checkpoints

| file | LoRA tensors | projection | size |
|---|---|---|---|
| prism-qwen3.5-9b-sft.pt | 256 | 4096×4096 bf16 | 266 MB |
| prism-qwen3.5-9b-grpo.pt | 256 | 4096×4096 bf16 | 266 MB |
| prism-gemma-2-9b-it-grpo.pt | 588 | 3584×3584 bf16 | 458 MB |
| prism-ministral-3-8b-grpo.pt | 476 | 4096×4096 bf16 | 390 MB |

LoRA state-dict keys are prefixed
`base_model.model.model.language_model.layers...` (PEFT over the
`AutoModelForImageTextToText`/`AutoModelForCausalLM` wrapper; Ministral's
`lora_target_modules` is a regex **string** restricting adapters to the text
stack).

## Security note

Release files are pickled `torch.save` archives — load only checkpoints you
trust, or from the SHA256-verified prism-eval download script.
