import random
import re

import numpy as np
import torch
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def _is_vlm_config(model_name: str) -> bool:
    """True if the HF config has a vision tower (VLM)."""
    config = AutoConfig.from_pretrained(model_name)
    return hasattr(config, "vision_config") or hasattr(config, "vision_tower_config")


def dequantize_fp8_linears(model: AutoModelForCausalLM, dtype: torch.dtype) -> None:
    """Replace finegrained-FP8 Linear layers with plain bf16 nn.Linear, in place.

    Some checkpoints (e.g. mistralai/Ministral-3-8B-Instruct-2512) ship as per-tensor
    FP8 (block_size=None). We cannot train through them: the FP8 matmul kernel has no
    autograd backward, and transformers' on-load dequantizer only handles block-wise
    (2D) scales. We dequantize the weights ourselves — W = W_fp8.float() * weight_scale_inv
    — and clear the quantization markers so PEFT/DDP treat the model as ordinary bf16.
    """
    from transformers.integrations.finegrained_fp8 import FP8Linear

    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, FP8Linear):
                continue
            w = (child.weight.data.to(torch.float32) * child.weight_scale_inv.data.to(torch.float32)).to(dtype)
            new = torch.nn.Linear(
                child.in_features, child.out_features, bias=child.bias is not None, dtype=dtype, device=w.device
            )
            new.weight.data.copy_(w)
            if child.bias is not None:
                new.bias.data.copy_(child.bias.data.to(dtype))
            setattr(parent, child_name, new)

    model.hf_quantizer = None
    model.is_quantized = False
    if hasattr(model.config, "quantization_config"):
        delattr(model.config, "quantization_config")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch for reproducible runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(
    model_name: str,
    dtype: torch.dtype,
    **model_kwargs,
) -> AutoModelForCausalLM:
    print("🧠 Loading model...")

    # Gemma prefers eager attention; others use FA2 when available, else SDPA.
    # flash-attn is ABI-locked to torch and may not import on newer torch builds.
    if "gemma" in model_name.lower():
        attn = "eager"
    else:
        try:
            import flash_attn  # noqa: F401

            attn = "flash_attention_2"
        except Exception:
            attn = "sdpa"

    kwargs: dict = {
        "device_map": "auto",
        "attn_implementation": attn,
        "torch_dtype": dtype,
        **model_kwargs,
    }

    if _is_vlm_config(model_name):
        model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    # FP8 checkpoints can't be trained through (no autograd backward on the FP8 matmul);
    # dequantize to bf16 so LoRA training/eval behaves like an ordinary model.
    quant_cfg = getattr(model.config, "quantization_config", None)
    if quant_cfg is not None:
        quant_method = quant_cfg.get("quant_method") if isinstance(quant_cfg, dict) else getattr(quant_cfg, "quant_method", None)
        quant_method = getattr(quant_method, "value", quant_method)  # unwrap QuantizationMethod enum
        if quant_method == "fp8":
            print("⚙️  Dequantizing FP8 linears to bf16 for training")
            dequantize_fp8_linears(model, dtype)
    return model


def load_tokenizer(
    model_name: str,
) -> AutoTokenizer:
    # Load tokenizer
    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    # Ministral/Mistral chat templates inject a ~550-token default system prompt when no
    # system message is provided. For oracle training/eval we want the subject model to see
    # raw content (parity with the Qwen/Gemma recipes, which inject nothing), and the bloat
    # blows past our 64GB dataset-load budget. Blank the default so no system block is emitted.
    template = getattr(tokenizer, "chat_template", None)
    if template and "default_system_message" in template:
        tokenizer.chat_template = re.sub(
            r"set default_system_message = '.*?'\s*%}",
            "set default_system_message = '' %}",
            template,
            count=1,
            flags=re.DOTALL,
        )

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if not tokenizer.bos_token_id:
        tokenizer.bos_token_id = tokenizer.eos_token_id
    return tokenizer


def list_decode(x: torch.Tensor, tokenizer: AutoTokenizer) -> list[list[str]]:
    """
    Input: torch.Tensor of shape [batch_size, seq_length]
    Output: list of list of strings of len [batch_size, seq_length] Each inner list corresponds to a single token
    """
    assert len(x.shape) == 1 or len(x.shape) == 2
    # Convert to list of lists, even if x is 1D
    if len(x.shape) == 1:
        x = x.unsqueeze(0)  # Make it 2D for consistent handling

    # Convert tensor to list of list of ints
    token_ids = x.tolist()

    # Convert token ids to token strings
    return [tokenizer.batch_decode(seq, skip_special_tokens=False) for seq in token_ids]


def get_bos_eos_pad_mask(tokenizer: AutoTokenizer, token_ids: torch.Tensor) -> torch.Tensor:
    """Create mask for BOS, EOS, and PAD tokens"""
    mask = torch.zeros_like(token_ids, dtype=torch.bool)

    if tokenizer.bos_token_id is not None:
        mask |= token_ids == tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        mask |= token_ids == tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        mask |= token_ids == tokenizer.pad_token_id

    return mask


def assert_no_peft_present(model, check_for_active_adapter_only=False):
    """
    Asserts that no PEFT adapters are present or active on the model.

    Args:
        model: The model to check.
        check_for_active_adapter_only (bool):
            - If False (default), asserts that NO adapters are loaded on the model at all.
            - If True, asserts only that no adapter is currently *active*.
              This allows inactive adapters to still be loaded in memory.
    """
    is_peft_model = isinstance(model, PeftModel)

    if not is_peft_model and not hasattr(model, "peft_config"):
        # If it's not a PeftModel and has no peft_config, we're 100% sure no adapters are loaded.
        return

    # At this point, the model has had PEFT adapters at some point.

    # getattr is used to safely access peft_config, which might be an empty dict.
    loaded_adapters = list(getattr(model, "peft_config", {}).keys())

    if not check_for_active_adapter_only:
        assert not loaded_adapters, (
            f"PEFT check failed! Found loaded adapters: {loaded_adapters}. "
            "Model should have no adapters loaded in memory."
        )

    # PeftModel has an `active_adapters` property which is a list of active adapter names.
    # It's an empty list when the base model is active.
    active_adapters = getattr(model, "active_adapters", [])
    assert not active_adapters, (
        f"PEFT check failed! Found active adapters: {active_adapters}. Model should be running in base mode."
    )


def get_layer_count(model_name: str) -> int:
    """Get the number of layers from a HuggingFace model config."""
    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    elif hasattr(config, "text_config"):
        # Gemma-3 models store config in text_config
        return config.text_config.num_hidden_layers
    raise AttributeError(f"Could not find layer count for {model_name}")


def layer_percent_to_layer(model_name: str, layer_percent: int) -> int:
    """Convert a layer percent to a layer number."""
    max_layers = get_layer_count(model_name)
    return int(max_layers * (layer_percent / 100))
