"""prism.rl — judge-driven GRPO training for the PRISM monitor.

Projection + LoRA monitor: raw hooked activations feed straight into a
trainable linear projection, then the LoRA-adapted target model decodes
the ITM report (same architecture the prism-eval runner loads).

Layout:
  config.py       RL_CONFIG inheriting FINETUNE_CONFIG
  judge.py   LLM-judge wrapper (per-bullet recall + hallucination)
  losses.py       grpo_loss / dpo_loss / ipo_loss + logprob helpers
  adapters.py     PEFT policy/ref LoRA pair + ProjectionPair
  rollouts.py     N-candidate sampled generation via inputs_embeds
  data.py         prefix + full-sequence embedding builders
  train.py        main training loop
"""
