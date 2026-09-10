"""Per-target-model profiles for the prism.sft / prism.rl pipeline.

Selected via the ``PRISM_TARGET_MODEL`` env var. When it is unset (or set to
``qwen3.5-9b``) every code path reproduces the original Qwen3.5-9B behaviour, so
existing runs/tests are unaffected.

Additional profiles: ``qwen3.5-0.8b`` (Qwen/Qwen3.5-0.8B, hook layer 12),
``gemma2-9b`` (google/gemma-2-9b-it, hook layer 21) and
``ministral3-8b`` (mistralai/Ministral-3-8B-Instruct-2512-BF16, hook layer 17).

Profile fields
--------------
model_id              HF repo id.
load_class            Auto class for SFT/RL loading: "causal_lm" | "image_text_to_text".
extract_load_class    Auto class for activation extraction (Qwen uses causal_lm here
                      even though training uses image_text_to_text — matches the
                      pre-existing precompute path).
tokenizer_via_processor  True → AutoProcessor(...).tokenizer (Qwen); False → AutoTokenizer.
hidden_size           Residual-stream width at the hook layer (== projection_dim).
hook_layer            Decoder layer index to hook (proportional mid-depth).
attn_implementation   None → omit kwarg (HF default sdpa); "eager" for gemma-2
                      (sdpa silently drops its attn/final logit softcapping → wrong
                      activations); "sdpa" for ministral.
turn_end_token        Token that ends an assistant turn if != eos (gemma: <end_of_turn>).
                      None → use tokenizer.eos_token_id (Qwen, Ministral).
system_message_override  None → no system message (Qwen, gemma — gemma's template
                      forbids a system role); "" → inject an empty system message to
                      suppress Ministral's ~530-token default system prompt.
lora_target_modules   None → keep the config's list; a regex string restricts LoRA to
                      the text stack (Ministral's pixtral vision tower shares proj names).
force_hf_tokenizer    True → bypass AutoTokenizer routing and load the HF
                      tokenizers backend (Ministral: mistral_common rejects
                      assistant-final conversations).
needs_qwen35_rope_patch  Apply the transformers 5.3 Qwen3_5 compute_3d_position_ids
                      monkeypatch (Qwen only).
tag                   Short label for checkpoint/dir/run naming.
"""

from __future__ import annotations

import os

import torch

_DEFAULT = "qwen3.5-9b"

# Ministral-3 is a Mistral3ForConditionalGeneration VLM: its pixtral vision tower
# reuses q/k/v/o/gate/up/down proj names, so a plain module list would attach dead
# LoRA adapters there. Restrict to any such proj that sits under `language_model`.
_MINISTRAL_LORA_REGEX = (
    r".*language_model\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)

PROFILES: dict[str, dict] = {
    "qwen3.5-9b": dict(
        model_id="Qwen/Qwen3.5-9B",
        load_class="image_text_to_text",
        extract_load_class="causal_lm",
        tokenizer_via_processor=True,
        hidden_size=4096,
        hook_layer=16,
        attn_implementation=None,
        turn_end_token=None,
        system_message_override=None,
        lora_target_modules=None,
        needs_qwen35_rope_patch=True,
        tag="qwen3.5-9b-L16",
    ),
    "qwen3.5-0.8b": dict(
        # Same Qwen3.5 architecture family as the 9B (model_type qwen3_5,
        # Qwen3_5ForConditionalGeneration), so the load classes, processor
        # tokenizer, and RoPE patch are identical — only the width, depth, and
        # tag differ. hidden_size 1024 (==projection_dim); 24 decoder layers, so
        # the mid-depth hook is layer 12 (== the 16/32 relative depth of the 9B).
        model_id="Qwen/Qwen3.5-0.8B",
        load_class="image_text_to_text",
        extract_load_class="causal_lm",
        tokenizer_via_processor=True,
        hidden_size=1024,
        hook_layer=12,                 # 12/24 = mid-depth (matches 16/32 on the 9B)
        attn_implementation=None,
        turn_end_token=None,
        system_message_override=None,
        lora_target_modules=None,
        needs_qwen35_rope_patch=True,
        tag="qwen3.5-0.8b-L12",
    ),
    "gemma2-9b": dict(
        model_id="google/gemma-2-9b-it",
        load_class="causal_lm",
        extract_load_class="causal_lm",
        tokenizer_via_processor=False,
        hidden_size=3584,
        hook_layer=21,                 # 21/42 ≈ 16/32 (Qwen relative depth)
        attn_implementation="eager",   # sdpa drops softcapping → wrong activations
        turn_end_token="<end_of_turn>",  # id 107; eos (1) never appears in turns
        system_message_override=None,
        lora_target_modules=None,      # gemma-2 has no vision tower; keep the list
        needs_qwen35_rope_patch=False,
        tag="gemma2-9b-L21",
    ),
    "ministral3-8b": dict(
        model_id="mistralai/Ministral-3-8B-Instruct-2512-BF16",
        load_class="image_text_to_text",   # mistral3 only in the ImageTextToText map
        extract_load_class="image_text_to_text",
        tokenizer_via_processor=False,
        # When mistral_common is installed (e.g. via the vllm extra),
        # AutoTokenizer routes to MistralCommonBackend, whose validator
        # rejects the assistant-final conversations activation extraction
        # tokenizes. The released runs used the HF tokenizers backend —
        # force it (verified token-identical to the released tokenization).
        force_hf_tokenizer=True,
        hidden_size=4096,
        hook_layer=17,                 # 17/34 ≈ 16/32
        attn_implementation="sdpa",
        turn_end_token=None,           # </s> == eos
        system_message_override="",    # suppress ~530-token default system prompt
        lora_target_modules=_MINISTRAL_LORA_REGEX,
        needs_qwen35_rope_patch=False,
        tag="ministral3-8b-L17",
    ),
}


def active_name() -> str:
    """The selected profile name (env PRISM_TARGET_MODEL, default qwen3.5-9b)."""
    return os.environ.get("PRISM_TARGET_MODEL") or _DEFAULT


def is_active() -> bool:
    """True when a non-Qwen target model is selected."""
    return active_name() != _DEFAULT


def get_profile(name: str | None = None) -> dict:
    """Return a copy of the selected (or named) profile."""
    key = name or active_name()
    if key not in PROFILES:
        raise KeyError(
            f"Unknown target model {key!r}. Known: {sorted(PROFILES)}"
        )
    return dict(PROFILES[key])


def _apply_qwen35_rope_patch() -> None:
    """Force Qwen3_5Model.compute_3d_position_ids → None (text-only workflow).

    See prism/rl/train.py for the rationale (batch-size-shrink bug in the
    cached rope_deltas path). Only applied for the Qwen profile.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5
    modeling_qwen3_5.Qwen3_5Model.compute_3d_position_ids = (
        lambda self, *args, **kwargs: None
    )


def _model_class(kind: str):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    return {
        "causal_lm": AutoModelForCausalLM,
        "image_text_to_text": AutoModelForImageTextToText,
    }[kind]


def load_tokenizer(profile: dict):
    """Load the text tokenizer for a profile (AutoTokenizer, or the Qwen processor)."""
    if profile["tokenizer_via_processor"]:
        from transformers import AutoProcessor
        tok = AutoProcessor.from_pretrained(
            profile["model_id"], trust_remote_code=True
        ).tokenizer
    elif profile.get("force_hf_tokenizer"):
        # Bypass Auto routing (see the profile comment): load the plain HF
        # tokenizers backend, matching the environment the released
        # checkpoints were trained in.
        try:
            from transformers import TokenizersBackend
            tok = TokenizersBackend.from_pretrained(
                profile["model_id"], trust_remote_code=True
            )
        except ImportError:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(
                profile["model_id"], trust_remote_code=True, fix_mistral_regex=True
            )
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            profile["model_id"], trust_remote_code=True
        )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_target_model(
    device,
    dtype=torch.bfloat16,
    *,
    for_extraction: bool = False,
    profile: dict | None = None,
):
    """Load (model, tokenizer, profile) for the selected target model.

    ``for_extraction`` selects ``extract_load_class`` (Qwen uses AutoModelForCausalLM
    for extraction but AutoModelForImageTextToText for training).
    """
    prof = profile or get_profile()
    if prof["needs_qwen35_rope_patch"]:
        _apply_qwen35_rope_patch()

    tok = load_tokenizer(prof)

    kind = prof["extract_load_class"] if for_extraction else prof["load_class"]
    kwargs = dict(dtype=dtype, device_map=device, trust_remote_code=True)
    if prof["attn_implementation"] is not None:
        kwargs["attn_implementation"] = prof["attn_implementation"]
    model = _model_class(kind).from_pretrained(prof["model_id"], **kwargs)
    return model, tok, prof


def resolve_gen_eos_id(profile: dict, tokenizer) -> int:
    """Token id that terminates generation (gemma: <end_of_turn>, else eos)."""
    tok_str = profile.get("turn_end_token")
    if tok_str is not None:
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if tid is not None and tid != tokenizer.unk_token_id:
            return tid
    return tokenizer.eos_token_id


def wrap_messages(messages: list[dict], profile: dict) -> list[dict]:
    """Prepend the profile's system-message override, if any and not already present."""
    ov = profile.get("system_message_override")
    if ov is not None and (not messages or messages[0].get("role") != "system"):
        return [{"role": "system", "content": ov}] + list(messages)
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# Config overlay
# ─────────────────────────────────────────────────────────────────────────────

def _env_int(name: str):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else None


def _env_float(name: str):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else None


def apply_profile_overlay(cfg: dict) -> dict:
    """Mutate a FINETUNE/RL config in place for the selected target model.

    No-op field changes when Qwen is selected (only generic env path overrides
    are applied). Called at the end of each module's config.py.
    """
    # Generic env overrides (apply for every profile, incl. Qwen).
    for env_key, cfg_key in [
        ("PRISM_PRECOMPUTED_DIR", "precomputed_dir"),
        ("PRISM_CHECKPOINT_DIR", "checkpoint_dir"),
        ("PRISM_WANDB_RUN_NAME", "wandb_run_name"),
        ("PRISM_SFT_INIT_FROM", "sft_init_from"),
        ("PRISM_VALID_RECORD_IDS", "valid_record_ids"),
    ]:
        val = os.environ.get(env_key)
        if val:
            cfg[cfg_key] = val
    # On-the-fly data: os.pathsep-separated JSONL paths.
    val = os.environ.get("PRISM_DATASET_PATHS")
    if val:
        cfg["dataset_paths"] = [p for p in val.split(os.pathsep) if p]
    for env_key, cfg_key in [
        ("PRISM_EVAL_EVERY", "eval_every"),
        ("PRISM_EVAL_SAMPLES", "eval_samples"),
        ("PRISM_MAX_OPT_STEPS", "max_opt_steps"),
        ("PRISM_BATCH_SIZE", "batch_size"),
        ("PRISM_GRAD_ACCUM", "grad_accum"),
        ("PRISM_LOG_EVERY", "log_every"),
        # Training-RNG seed (rollout sampling, data order, init). NOTE: with
        # precomputed activations the train/val split is frozen at extraction
        # time, so this varies training stochasticity only; in the on-the-fly
        # SFT path it ALSO changes the train/val split.
        ("PRISM_SEED", "seed"),
    ]:
        iv = _env_int(env_key)
        if iv is not None:
            cfg[cfg_key] = iv
            if cfg_key == "eval_samples":
                cfg["_eval_samples_explicit"] = iv
    # Float overrides — used by recipes to pin the released SFT learning rates
    # exactly (they were sweep-sampled, so not round numbers).
    for env_key, cfg_key in [
        ("PRISM_LR", "lr"),
        ("PRISM_PROJECTION_LR", "projection_lr"),
    ]:
        fv = _env_float(env_key)
        if fv is not None:
            cfg[cfg_key] = fv

    # Cap the SFT target (instruction_set) length. Large-vocab targets make the
    # LM-head cross-entropy the memory bottleneck (batch x seq x vocab, fp32-
    # upcast), so bounding seq lets a much larger batch fit. Only takes effect
    # where the config has the key (SFT), so RL is unaffected.
    mt = _env_int("PRISM_MAX_TARGET_LEN")
    if mt is not None and "max_target_len" in cfg:
        cfg["max_target_len"] = mt

    name = active_name()
    cfg["_target_profile"] = name
    if name == _DEFAULT:
        return apply_hook_layer_override(cfg)

    prof = PROFILES[name]
    cfg["model_id"] = prof["model_id"]
    cfg["hook_layer"] = prof["hook_layer"]
    cfg["projection_dim"] = prof["hidden_size"]
    if prof["lora_target_modules"] is not None:
        cfg["lora_target_modules"] = prof["lora_target_modules"]
    # gemma-2 has a 256k vocab; shrink the token-exact KL chunk as insurance
    # (production runs use k3, which never materialises full-vocab KL).
    if name == "gemma2-9b" and "kl_chunk_size" in cfg:
        cfg["kl_chunk_size"] = min(cfg["kl_chunk_size"], 128)
    return apply_hook_layer_override(cfg)


def apply_hook_layer_override(cfg: dict) -> dict:
    """Apply the PRISM_HOOK_LAYER env override (layer-selection ablations).

    Called from apply_profile_overlay AFTER profile pinning so the override
    wins for every profile. No-op when the env var is unset.
    """
    hv = _env_int("PRISM_HOOK_LAYER")
    if hv is not None:
        cfg["hook_layer"] = hv
    return cfg

