"""
model.py — Model components for activation-conditioned decoder finetuning.

Components:
  1. ActivationProjection — Linear(hidden→hidden), trainable, maps hooked
     activations into the decoder embedding space
  2. Target model with LoRA — single instance, LoRA toggled on/off for extraction vs decoding

The key trick: one target-model instance serves both roles:
  - LoRA OFF + inference_mode: activation extraction (hook at layer 16, EarlyExit)
  - LoRA ON + gradients: decoder for teacher-forced generation
"""

import logging
from typing import Optional
from collections import OrderedDict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Projection
# ─────────────────────────────────────────────────────────────────────────────

class ActivationProjection(nn.Module):
    """
    Linear projection from activation space → decoder embedding space.
    Trained from scratch during finetuning.
    """

    def __init__(self, dim: int = 4096):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# Norm matching
# ─────────────────────────────────────────────────────────────────────────────

def compute_target_norm(model) -> torch.Tensor:
    """Compute mean embedding norm from the decoder's embedding table."""
    with torch.no_grad():
        emb_weight = model.get_input_embeddings().weight
        return emb_weight.norm(dim=1).mean()


def norm_match(soft_tokens: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    """
    Scale soft tokens to match the decoder's embedding norm.
    soft_tokens: [B, N, D] or [N, D]
    """
    norms = soft_tokens.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return soft_tokens / norms * target_norm


# ─────────────────────────────────────────────────────────────────────────────
# Activation hook + EarlyExit
# ─────────────────────────────────────────────────────────────────────────────

class _EarlyExit(Exception):
    """Raised in the forward hook to abort the target model's forward pass after the hooked layer."""


def _find_layers(model):
    """Find the transformer layer list in the model."""
    best = None

    def _walk(module, prefix=""):
        nonlocal best
        for name, child in module._modules.items():
            if child is None:
                continue
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.ModuleList) and len(child) > 1:
                types = {type(c).__name__ for c in child}
                if len(types) == 1:
                    if best is None or len(child) > len(best[1]):
                        best = (path, child)
            _walk(child, path)

    _walk(model)
    if best is None:
        raise RuntimeError("Cannot locate transformer layers in model.")

    return best[1]


def register_hook(model, layer_idx: int):
    """
    Register a forward hook on layers[layer_idx] that captures hidden states
    and raises _EarlyExit to skip remaining layers + lm_head.

    Returns:
        activation_store: dict with key "hidden" populated after each call
        hook_handle: call .remove() to deregister
    """
    layers = _find_layers(model)
    assert 0 <= layer_idx < len(layers), (
        f"hook_layer={layer_idx} out of range for model with {len(layers)} layers."
    )

    activation_store = {"active": False}

    def _hook(module, inp, out):
        if not activation_store["active"]:
            return  # Hook disabled during decoding — let the full forward pass run
        hidden = out[0] if isinstance(out, tuple) else out
        activation_store["hidden"] = hidden.detach()
        raise _EarlyExit()

    handle = layers[layer_idx].register_forward_hook(_hook)
    logger.info(f"Hook registered on layer {layer_idx}/{len(layers)-1}.")
    return activation_store, handle


# ─────────────────────────────────────────────────────────────────────────────
# Activation extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_activations(
    target_model,
    act_store: dict,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_only_lens: torch.Tensor,
    response_token_counts: torch.Tensor,
    max_act_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the target model in inference mode (LoRA disabled externally), extract activations
    from the response portion.

    Args:
        target_model:           the target model (LoRA should be disabled before calling)
        act_store:            dict populated by the hook with key "hidden"
        input_ids:            [B, seq_len] — tokenized prompt + response
        attention_mask:       [B, seq_len]
        prompt_only_lens:     [B] — token count of prompt alone per sample
        response_token_counts: [B] — actual activation count to extract per sample (≤ max_act_tokens)
        max_act_tokens:       maximum activation tokens

    Returns:
        act_padded:  [B, max_N, D] — padded activation sequences (bfloat16)
        act_mask:    [B, max_N]    — 1=valid, 0=pad
    """
    B = input_ids.shape[0]
    device = input_ids.device

    act_store["active"] = True
    with torch.inference_mode():
        try:
            target_model(input_ids=input_ids, attention_mask=attention_mask)
        except _EarlyExit:
            pass
    act_store["active"] = False

    hidden = act_store["hidden"]  # [B, seq_len, D]
    D = hidden.shape[-1]

    # Extract tail of response activations per sample
    max_N = int(response_token_counts.max().item())
    max_N = min(max_N, max_act_tokens)

    act_padded = torch.zeros(B, max_N, D, dtype=hidden.dtype, device=device)
    act_mask = torch.zeros(B, max_N, dtype=torch.long, device=device)

    for b in range(B):
        total_len = int(attention_mask[b].sum().item())
        prompt_len = int(prompt_only_lens[b].item())
        resp_len = total_len - prompt_len
        n_act = min(resp_len, max_act_tokens)

        if n_act <= 0:
            continue

        # Take last n_act activations from the response portion
        start = total_len - n_act
        act_padded[b, :n_act, :] = hidden[b, start:total_len, :]
        act_mask[b, :n_act] = 1

    return act_padded, act_mask
