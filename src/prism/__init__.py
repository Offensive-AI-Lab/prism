"""PRISM training pipeline.

Trains activation-reading monitor models for
"PRISM: Recovering Instruction Sets from Language Model Activations"
(arXiv:2606.09563). Evaluation lives in the sibling repo
https://github.com/Offensive-AI-Lab/prism-eval.

Subpackages:
    datagen      -- oracle dataset generation and filtering
    activations  -- offline activation precompute (sharded safetensors)
    sft          -- supervised finetuning of the monitor (projection + LoRA)
    rl           -- GRPO with an LLM-judge reward on top of the SFT checkpoint
    calibration  -- judge-vs-human calibration tooling (kappa/ICC scoring)
"""

__version__ = "1.0.0"
