"""data.py — embedding-sequence builders for the projection variant.

Two builders, matching the two loss families:

  * build_prefix_embeddings(...)  — once per outer step. Produces the
    [scaled soft tokens | prompt_b token embeds] prefix that feeds
    `target_model.generate(inputs_embeds=...)`. Used by rollouts and by the
    loss-time forward.

  * build_group_inputs(...) / build_pair_inputs(...) — once per
    candidate. Concatenates the prefix with the candidate's token
    embeddings to produce the full-sequence `inputs_embeds` + labels
    for the policy/ref logp forwards.

The label convention is the same as
in `prism/sft/train.py`: -100 on the prefix
positions (soft tokens + prompt_b template), candidate token ids on
the response span. The cross-entropy shift inside the model handles the
+1 offset automatically.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def build_prefix_embeddings(
    soft_scaled: torch.Tensor,         # [B, N_act_max, D]
    act_mask: torch.Tensor,            # [B, N_act_max]
    chat_prefix_input_ids: torch.Tensor,  # [B, P]
    chat_prefix_attention_mask: torch.Tensor,  # [B, P]
    target_model: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate scaled soft tokens + prompt_b token embeds into a single
    `inputs_embeds` prefix per sample, padded to a common length.

    Returns:
        prefix_embeds:   [B, max_prefix_len, D]   left-padded with zeros
        prefix_mask:     [B, max_prefix_len]      1 = real token, 0 = pad
        prefix_lengths:  [B]                       per-sample real length
    """
    device = soft_scaled.device
    B = soft_scaled.size(0)
    D = soft_scaled.size(-1)

    embed_layer = target_model.get_input_embeddings()
    chat_embeds_full = embed_layer(chat_prefix_input_ids)  # [B, P, D]

    seqs: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    lengths: list[int] = []
    for b in range(B):
        n_act = int(act_mask[b].sum().item())
        p_len = int(chat_prefix_attention_mask[b].sum().item())
        # Slicing soft_scaled[:n_act] assumes the activation mask is right-aligned
        # (real tokens at [:n_act], pad at [n_act:]). A left-aligned or
        # non-contiguous mask would silently splice in zero rows instead of real
        # activations. Cheap assert to lock that contract in.
        assert bool(act_mask[b, :n_act].all()) and not bool(act_mask[b, n_act:].any()), (
            f"build_prefix_embeddings expects right-aligned act_mask; "
            f"got non-contiguous mask at b={b}"
        )
        soft_b = soft_scaled[b, :n_act, :]
        chat_b = chat_embeds_full[b, :p_len, :]
        seq_emb = torch.cat([soft_b, chat_b], dim=0)
        seq_mask = torch.ones(seq_emb.size(0), device=device, dtype=chat_prefix_attention_mask.dtype)
        seqs.append(seq_emb)
        masks.append(seq_mask)
        lengths.append(seq_emb.size(0))

    # Pad to batch max. Use LEFT padding because generate() with
    # inputs_embeds is left-padding-aware; right-padding would push the
    # generation cursor into the pad region.
    max_len = max(lengths)
    prefix_embeds = torch.zeros(B, max_len, D, device=device, dtype=soft_scaled.dtype)
    prefix_mask = torch.zeros(B, max_len, device=device, dtype=chat_prefix_attention_mask.dtype)
    for b, (seq, m) in enumerate(zip(seqs, masks)):
        L = seq.size(0)
        prefix_embeds[b, max_len - L :, :] = seq
        prefix_mask[b, max_len - L :] = m

    return prefix_embeds, prefix_mask, torch.tensor(lengths, device=device, dtype=torch.long)


def build_full_inputs(
    soft_scaled: torch.Tensor,         # [B, N_act_max, D]
    act_mask: torch.Tensor,            # [B, N_act_max]
    chat_prefix_input_ids: torch.Tensor,  # [B, P]
    chat_prefix_attention_mask: torch.Tensor,  # [B, P]
    candidate_token_ids: list[list[int]],   # length B  — per-prompt candidate text token ids
    target_model: nn.Module,
    pad_id: int,
) -> dict:
    """Build full-sequence `inputs_embeds` + labels for the loss-time forward.

    Sequence layout per sample:
        [scaled_soft (n_act) | prompt_b template (p_len) | candidate (R)]
        └── labels = -100 ───┘└──── labels = -100 ──────┘└─ labels = ids ┘

    Returns a dict with:
        inputs_embeds:     [B, T, D]   right-padded with zero embeddings
        attention_mask:    [B, T]
        labels:            [B, T]      -100 everywhere outside the response span
        response_lengths:  [B]         per-sample response token count (for diagnostics)
    """
    device = soft_scaled.device
    B = soft_scaled.size(0)
    D = soft_scaled.size(-1)
    assert len(candidate_token_ids) == B

    embed_layer = target_model.get_input_embeddings()
    chat_embeds_full = embed_layer(chat_prefix_input_ids)  # [B, P, D]

    seqs: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    resp_lens: list[int] = []

    for b in range(B):
        n_act = int(act_mask[b].sum().item())
        p_len = int(chat_prefix_attention_mask[b].sum().item())
        cand_ids = candidate_token_ids[b]
        cand_t = torch.tensor(cand_ids, dtype=torch.long, device=device)
        cand_embeds = embed_layer(cand_t)  # [R, D]
        R = cand_embeds.size(0)

        soft_b = soft_scaled[b, :n_act, :]
        chat_b = chat_embeds_full[b, :p_len, :]

        seq_emb = torch.cat([soft_b, chat_b, cand_embeds], dim=0)   # [n_act + p_len + R, D]
        L = seq_emb.size(0)
        seq_mask = torch.ones(L, device=device, dtype=chat_prefix_attention_mask.dtype)
        seq_labels = torch.full((L,), -100, dtype=torch.long, device=device)
        # Response span starts at n_act + p_len.
        seq_labels[n_act + p_len : n_act + p_len + R] = cand_t

        seqs.append(seq_emb)
        masks.append(seq_mask)
        label_rows.append(seq_labels)
        resp_lens.append(R)

    # Right-pad to batch max for the loss-time forward (causal LM, mask handles padding).
    max_len = max(s.size(0) for s in seqs)
    inputs_embeds = torch.zeros(B, max_len, D, device=device, dtype=soft_scaled.dtype)
    attention_mask = torch.zeros(B, max_len, device=device, dtype=chat_prefix_attention_mask.dtype)
    labels = torch.full((B, max_len), -100, dtype=torch.long, device=device)
    for b, (seq, m, lab) in enumerate(zip(seqs, masks, label_rows)):
        L = seq.size(0)
        inputs_embeds[b, :L, :] = seq
        attention_mask[b, :L] = m
        labels[b, :L] = lab

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "labels": labels,
        "response_lengths": torch.tensor(resp_lens, device=device, dtype=torch.long),
    }


# ─── Group / pair helpers (B prompts × N candidates → flat tensors) ──────────

def expand_for_group(
    soft_scaled: torch.Tensor,         # [B, N_act, D]
    act_mask: torch.Tensor,            # [B, N_act]
    chat_prefix_input_ids: torch.Tensor,
    chat_prefix_attention_mask: torch.Tensor,
    n_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repeat per-prompt tensors `n_candidates` times to flatten to [B*N, ...].

    Returns (soft_scaled_rep, act_mask_rep, chat_ids_rep, chat_mask_rep, group_ids).
    """
    N = n_candidates
    B = soft_scaled.size(0)
    device = soft_scaled.device

    return (
        soft_scaled.repeat_interleave(N, dim=0),
        act_mask.repeat_interleave(N, dim=0),
        chat_prefix_input_ids.repeat_interleave(N, dim=0),
        chat_prefix_attention_mask.repeat_interleave(N, dim=0),
        torch.arange(B, device=device, dtype=torch.long).repeat_interleave(N),
    )


def select_pair_from_group(
    scores: Sequence[float],
    min_margin: float = 0.0,
) -> tuple[int, int] | None:
    """Return (argmax_idx, argmin_idx) if margin >= min_margin, else None.

    Used by DPO/IPO to collapse an N-candidate judged group into one preference
    pair. None signals "skip this prompt this round" (ties / low-margin).
    """
    if len(scores) < 2:
        return None
    best = max(range(len(scores)), key=lambda i: scores[i])
    worst = min(range(len(scores)), key=lambda i: scores[i])
    if best == worst:
        return None
    if scores[best] - scores[worst] < min_margin:
        return None
    return best, worst
