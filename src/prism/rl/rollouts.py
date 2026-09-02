"""rollouts.py — sample N candidates per prompt via `inputs_embeds`.

Mirrors the validation generate path from
`prism/sft/train.py`. No layer-0 hook, no KV
prefill trick — the activations enter as scaled soft tokens in the
embedding prefix, and the target model's generate() handles the rest natively.

Caller responsibilities:
  - Provide `prefix_embeds` already containing
    `[scaled soft tokens | retrieval_prompt token embeds]` and a matching
    attention mask. (Building those is `prism/rl/data.py`.)
  - Ensure the policy LoRA adapter is active and `target_model` is in
    eval mode for generation.

Returns BOTH the decoded texts (for the judge) and the actual sampled
token IDs (for the loss-time forward). The latter avoids the
detokenize→retokenize round trip which can silently produce a different
token sequence than what the policy sampled — and also preserves the
EOS token the policy chose to emit, so the loss can score that decision.
"""

from __future__ import annotations

from typing import Optional

import torch


@torch.no_grad()
def generate_candidates(
    target_model,
    tokenizer,
    prefix_embeds: torch.Tensor,        # [B, P, D]
    prefix_attention_mask: torch.Tensor,  # [B, P]
    n_candidates: int,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.95,
    do_sample: bool = True,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
) -> tuple[list[list[str]], list[list[list[int]]]]:
    """Return ``(texts, token_ids)`` where each is indexed ``[b][i]``.

    For each of the B prompts, runs `n_candidates` independent samples
    through `target_model.generate(inputs_embeds=...)`. Sampling diversity comes
    from temperature + top_p; each sample gets a different RNG draw
    inside HuggingFace generate.

    - ``texts[b][i]`` — special-tokens-stripped decode, used by the judge.
    - ``token_ids[b][i]`` — the raw sampled token IDs (including the
      first EOS the policy emitted, if any; post-EOS pad tokens are
      trimmed). These are what the loss-time forward should embed,
      not a re-tokenization of ``texts``.
    """
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id  # some target models (e.g. Qwen) ship no pad token

    B, P, D = prefix_embeds.shape
    N = n_candidates

    # Repeat each row N times along batch.
    rep_embeds = prefix_embeds.repeat_interleave(N, dim=0)      # [B*N, P, D]
    rep_mask = prefix_attention_mask.repeat_interleave(N, dim=0)  # [B*N, P]

    gen_ids = target_model.generate(
        inputs_embeds=rep_embeds,
        attention_mask=rep_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        use_cache=True,
    )

    # When generate() is called with inputs_embeds (no input_ids), HF only
    # returns the NEWLY generated token ids (no prefix prepended).
    #
    # Trim post-EOS padding while KEEPING the first EOS — the model may have no
    # separate pad token, so generate fills with eos_token_id after the
    # sequence terminates. The first eos_id in the row is the actually
    # sampled stop token; we want it in the loss so the policy gets a
    # gradient signal on its termination decision.
    eos = int(eos_token_id) if eos_token_id is not None else None
    rows = gen_ids.tolist()
    trimmed: list[list[int]] = []
    for row in rows:
        if eos is not None and eos in row:
            cut = row.index(eos) + 1
            trimmed.append(row[:cut])
        else:
            trimmed.append(list(row))

    texts = tokenizer.batch_decode(trimmed, skip_special_tokens=True)
    texts_by_prompt = [texts[b * N : (b + 1) * N] for b in range(B)]
    ids_by_prompt = [trimmed[b * N : (b + 1) * N] for b in range(B)]
    return texts_by_prompt, ids_by_prompt
