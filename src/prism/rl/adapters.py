"""adapters.py — policy / ref pairs for LoRA AND projection.

Two things in the policy need a frozen reference twin for the KL term:

  1. LoRA on the decoder. Handled by PEFT's named-adapter machinery —
     `policy` adapter is trainable, `ref` adapter is frozen, both init
     from the SFT `lora_state`. Adapter swap via `set_adapter(name)`.

  2. `ActivationProjection` (Linear 4096→4096). Handled by a
     `ProjectionPair`: two `ActivationProjection` instances, both init
     from the SFT `projection_state`. The policy instance is trainable;
     the ref instance is frozen.

Workflow during training:
    pol_scaled = norm_match(pair.policy(act), target_norm)   # grad
    with torch.no_grad():
        ref_scaled = norm_match(pair.ref(act), target_norm)

    with adapter_scope(model, "policy"):
        pi_logp  = sequence_logprobs(model(... pol_scaled ...), labels)
    with adapter_scope(model, "ref"), torch.no_grad():
        ref_logp = sequence_logprobs(model(... ref_scaled ...), labels)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, set_peft_model_state_dict

logger = logging.getLogger(__name__)

POLICY_ADAPTER = "policy"
REF_ADAPTER = "ref"


def _load_sft_state(ckpt_path: str, key: str, device: str) -> dict:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ck.get(key)
    if state is None:
        raise KeyError(f"Checkpoint {ckpt_path} has no '{key}' key — needed for SFT init.")
    return state


def attach_policy_and_ref(
    target_model,
    lora_cfg: LoraConfig,
    sft_init_from: str,
    device: str,
) -> PeftModel:
    """Attach `policy` (trainable) + `ref` (frozen) LoRA adapters, both initialised
    from the same SFT checkpoint.

    Returns the wrapped PEFT model. The caller should treat this as their
    `target_model` going forward; the underlying base is unchanged.
    """
    sft_state = _load_sft_state(sft_init_from, "lora_state", device)

    # 1) wrap with the policy adapter and load SFT weights
    peft_model = get_peft_model(target_model, lora_cfg, adapter_name=POLICY_ADAPTER)
    set_peft_model_state_dict(peft_model, sft_state, adapter_name=POLICY_ADAPTER)
    logger.info("Loaded SFT LoRA state into adapter '%s' from %s", POLICY_ADAPTER, sft_init_from)

    # 2) add the ref adapter and load the SAME SFT state into it
    peft_model.add_adapter(REF_ADAPTER, lora_cfg)
    set_peft_model_state_dict(peft_model, sft_state, adapter_name=REF_ADAPTER)
    logger.info("Initialised ref adapter '%s' from the same SFT state", REF_ADAPTER)

    # 3) freeze ref params; policy stays trainable
    for name, p in peft_model.named_parameters():
        if f".{REF_ADAPTER}." in name:
            p.requires_grad_(False)

    # 4) start in policy mode
    peft_model.set_adapter(POLICY_ADAPTER)
    return peft_model


class adapter_scope:
    """Context manager that swaps the active adapter for the duration of a block.

    Usage:
        with adapter_scope(model, "ref"):
            with torch.no_grad():
                ref_out = model(...)
        # model is restored to the previously-active adapter
    """

    def __init__(self, model: PeftModel, adapter_name: str):
        self.model = model
        self.target = adapter_name
        self.prev: Optional[str] = None

    def __enter__(self):
        # PEFT stores active adapter on the base model
        active = getattr(self.model, "active_adapter", None)
        if isinstance(active, list) and active:
            self.prev = active[0]
        else:
            self.prev = active
        self.model.set_adapter(self.target)
        return self.model

    def __exit__(self, exc_type, exc, tb):
        if self.prev is not None:
            self.model.set_adapter(self.prev)
        return False


# ─── Projection pair ─────────────────────────────────────────────────────────

class ProjectionPair(torch.nn.Module):
    """Holds policy + ref instances of `ActivationProjection`.

    Both init from the same SFT `projection_state`. The policy module is
    trainable; the ref module is frozen and never updated.

    Callers pick which copy to use per forward — the pair is just storage,
    not a forward dispatcher.
    """

    def __init__(self, dim: int, sft_init_from: str, device: str, dtype=torch.bfloat16):
        super().__init__()
        from prism.sft.model import ActivationProjection

        sft_state = _load_sft_state(sft_init_from, "projection_state", device)

        self.policy = ActivationProjection(dim=dim).to(device=device, dtype=dtype)
        self.ref = ActivationProjection(dim=dim).to(device=device, dtype=dtype)

        # Load SFT weights into BOTH instances (strict to catch shape mismatches early).
        self.policy.load_state_dict(sft_state, strict=True)
        self.ref.load_state_dict(sft_state, strict=True)
        logger.info(
            "ProjectionPair: loaded SFT projection_state into both policy and ref (dim=%d)",
            dim,
        )

        # Freeze ref permanently. Policy stays trainable.
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.ref.eval()

    def trainable_parameters(self):
        """Convenience: parameters that should go into the optimiser."""
        return [p for p in self.policy.parameters() if p.requires_grad]
