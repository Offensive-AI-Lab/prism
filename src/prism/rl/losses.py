"""losses.py — GRPO / DPO / IPO losses + shared log-prob utilities.

Conventions:
  * `logits`: [B, T, V] from the decoder (full sequence, no shift).
  * `labels`: [B, T] aligned with logits; positions with label != -100 contribute.
    Caller is responsible for setting -100 outside the response span.
  * `sequence_logprobs(..., length_normalize=False)` returns a per-sequence SUM
    over response positions (DPO paper default); `length_normalize=True`
    returns the per-response-token mean (recommended for GRPO — see the
    historical notes §1.4 in the paper appendix).

GRPO works on groups of N candidates per prompt. The training script flattens
to [G*N] before calling and passes `group_ids` so we can normalise within group.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ─── Log-prob helpers ─────────────────────────────────────────────────────────

def sequence_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    length_normalize: bool = False,
    chunk_size: int | None = None,
    return_per_token: bool = False,
) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Per-sequence log-prob of `labels` under `logits`.

    If `length_normalize=False` (default), returns the SUM over response positions
    (DPO/IPO convention, kept for backwards-compat).
    If `length_normalize=True`, returns the MEAN per response token — required
    for GRPO so the policy-gradient signal isn't dominated by candidate length
    (a 200-token candidate's summed logp is ~10× a 20-token one, swamping the
    reward-driven advantage).

    `chunk_size` (optional): process the batch dim in chunks of this size to
    cap peak fp32 memory from `log_softmax`. The fp32 cast of full-vocab
    logits is the dominant transient at large vocabs (Qwen3.5: 152K) — at N=8 with
    [16, T, V] inputs, the cast peaks at ~10 GB and OOMs the 95 GB GPU.
    log_softmax is row-independent in the batch dim, so chunking is exact.
    Default `None` = no chunking (original behaviour).

    `return_per_token` (optional): if True, returns a 3-tuple
    `(seq_logp, token_lp, mask)` where token_lp is the per-token log-prob
    `[B, T-1]` (fp32, masked positions = 0) and mask is `[B, T-1]` bool. Used
    by callers that need per-token quantities (e.g. the k3 KL estimator) —
    same fp32 cast / gather as the aggregate path, so no extra compute.

    Returns [B] in fp32 — or the 3-tuple above when `return_per_token=True`.
    """
    # Standard causal-LM shift: position t predicts token t+1.
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    mask = (shift_labels != -100)
    # Replace -100 with 0 to make gather safe; the mask zeros it out.
    safe_labels = shift_labels.masked_fill(~mask, 0)

    B = shift_logits.shape[0]
    if chunk_size is None or chunk_size >= B:
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        token_lp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_lp = token_lp * mask.float()
    else:
        chunks = []
        for i in range(0, B, chunk_size):
            cl = shift_logits[i:i + chunk_size]
            csl = safe_labels[i:i + chunk_size]
            cm = mask[i:i + chunk_size]
            clp = F.log_softmax(cl.float(), dim=-1)
            chunk_tok_lp = clp.gather(-1, csl.unsqueeze(-1)).squeeze(-1)
            chunks.append(chunk_tok_lp * cm.float())
        token_lp = torch.cat(chunks, dim=0)

    summed = token_lp.sum(dim=-1)
    if length_normalize:
        counts = mask.float().sum(dim=-1).clamp_min(1.0)
        seq = summed / counts
    else:
        seq = summed

    if return_per_token:
        return seq, token_lp, mask
    return seq


def k3_kl_mean(
    pi_token_logp: torch.Tensor,
    ref_token_logp: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Schulman k3 KL(pi || ref) estimator, averaged across response tokens.

    Per response token, with ratio r = ref(x) / pi(x):
        k3(x) = (r - 1) - log(r)
              = exp(log_ref - log_pi) - 1 - (log_ref - log_pi)

    Properties:
      * Non-negative by construction (`exp(y) - 1 - y >= 0` for all real y).
      * Unbiased estimator of KL(pi || ref) under sampling from pi.
      * Low variance — strictly lower than the k1 estimator
        `(log_pi - log_ref).mean()` which can be negative.
      * Uses ONLY the sampled-token log-probs (no full-vocab softmax). At
        Qwen3.5's 152K vocab this saves ~3–5 GB peak memory per loss step
        vs the exact token-KL path — what lets us scale N past 4 on a
        95 GB GPU.

    Standard in production GRPO (TRL, OpenRLHF, DeepSeekMath).

    Inputs:
      pi_token_logp:  [B, T-1] fp32, grad ON  (policy per-token log-prob)
      ref_token_logp: [B, T-1] fp32, grad OFF (reference per-token log-prob)
      mask:           [B, T-1] bool — True at response positions

    Returns 0-d fp32 scalar.
    """
    log_ratio = ref_token_logp.detach() - pi_token_logp  # log(ref/pi)
    # Zero out masked positions so they don't pollute exp/log.
    log_ratio = log_ratio.masked_fill(~mask, 0.0)
    ratio = log_ratio.exp()
    k3 = (ratio - 1.0) - log_ratio  # >= 0 by exp inequality
    k3 = k3 * mask.float()
    n = mask.float().sum().clamp_min(1.0)
    return k3.sum() / n


def response_token_kl_mean(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Exact KL(policy || ref) averaged across response tokens in the batch.

    Per-token, exact-vocab KL between two categorical distributions —
    non-negative by construction (the sequence-level Monte-Carlo proxy
    `(pi_logp - ref_logp).mean()` can be negative on individual samples and
    silently rewards divergence). Memory-cheap because it gathers response
    positions before the fp32 softmax (otherwise the [B, T, V] fp32 cast can
    OOM at full vocab).

    `chunk_size` (optional): further chunk the gathered response tokens to
    cap peak fp32 memory. At N=8 with long candidates, the gathered
    [N_resp, V] fp32 tensor can still reach ~3 GB; with chunk_size=256
    it stays under ~1 GB. Default `None` = no chunking. The KL contributions
    are summed and then divided by total tokens, so chunking is exact.

    Returns a 0-d fp32 scalar: mean per-response-token KL over the batch.
    """
    shift_labels = labels[..., 1:].contiguous()
    mask = shift_labels != -100  # [B, T-1] bool
    if mask.sum() == 0:
        return policy_logits.new_zeros((), dtype=torch.float32)

    shift_pi = policy_logits[..., :-1, :]
    shift_ref = ref_logits[..., :-1, :]

    # Gather to [N_resp, V] *before* the fp32 cast so we never materialise
    # the full [B, T, V] tensor in fp32. ref_logits is no-grad (caller path);
    # policy gather preserves autograd into policy_logits.
    pi_resp = shift_pi[mask]
    ref_resp = shift_ref[mask].detach()

    N_resp = pi_resp.shape[0]
    if chunk_size is None or chunk_size >= N_resp:
        pi_logp = F.log_softmax(pi_resp.float(), dim=-1)
        ref_logp = F.log_softmax(ref_resp.float(), dim=-1)
        pi_prob = pi_logp.exp()
        kl_per_tok = (pi_prob * (pi_logp - ref_logp)).sum(dim=-1)  # [N_resp]
        return kl_per_tok.mean()

    kl_sum = pi_resp.new_zeros((), dtype=torch.float32)
    for i in range(0, N_resp, chunk_size):
        pi_c = pi_resp[i:i + chunk_size]
        ref_c = ref_resp[i:i + chunk_size]
        pi_lp = F.log_softmax(pi_c.float(), dim=-1)
        ref_lp = F.log_softmax(ref_c.float(), dim=-1)
        pi_p = pi_lp.exp()
        kl_chunk = (pi_p * (pi_lp - ref_lp)).sum(dim=-1)  # [chunk]
        kl_sum = kl_sum + kl_chunk.sum()
    return kl_sum / float(N_resp)


def sequence_token_kl(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Per-sequence KL(policy || ref) summed over response positions.

    Kept for compatibility — prefer `response_token_kl_mean` for GRPO. Same
    math, but materialises the full [B, T, V] fp32 tensor (expensive at
    Qwen3.5's 152K vocab size).
    """
    shift_pi = policy_logits[..., :-1, :].contiguous().float()
    shift_ref = ref_logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    mask = (shift_labels != -100).float()

    pi_logp = F.log_softmax(shift_pi, dim=-1)
    ref_logp = F.log_softmax(shift_ref, dim=-1)
    pi_prob = pi_logp.exp()
    kl_per_pos = (pi_prob * (pi_logp - ref_logp)).sum(dim=-1)
    kl_per_pos = kl_per_pos * mask
    return kl_per_pos.sum(dim=-1)  # [B]


# ─── GRPO ─────────────────────────────────────────────────────────────────────

def grpo_loss(
    policy_logp: torch.Tensor,   # [N] per-sequence logp (sum or mean — see length_normalize)
    ref_logp: torch.Tensor,      # [N] same convention as policy_logp, no_grad
    rewards: torch.Tensor,       # [N]
    group_ids: torch.Tensor,     # [N] int — same id = same prompt group
    kl_coef: float = 0.04,
    *,
    # Token-exact KL inputs (full-vocab path; memory-heavy)
    policy_logits: torch.Tensor | None = None,
    ref_logits: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    kl_chunk_size: int | None = None,
    # k3 KL inputs (memory-light, sampled-token-only path)
    pi_token_logp: torch.Tensor | None = None,
    ref_token_logp: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """Single-update GRPO (REINFORCE with group-relative baseline + KL).

    Advantage for candidate i with prompt g:
        adv_i = (r_i - mean_g(r)) / (std_g(r) + eps)

    Loss:
        L = -E[adv * logp_pi] + kl_coef * KL_term

    KL_term is selected by what the caller passes:
      * If `policy_logits`, `ref_logits`, `labels` are all provided →
        exact per-response-token KL(pi || ref) via response_token_kl_mean
        (non-negative, low variance, length-normalised by construction).
      * Otherwise → legacy sequence-level MC proxy `(pi_logp - ref_logp).mean()`,
        which can go negative on individual samples and silently reward
        divergence. Kept only for the test suite + DPO/IPO-style callers; do
        NOT use for production GRPO. See the historical notes §1.4.
    """
    device = policy_logp.device
    rewards = rewards.to(device=device, dtype=torch.float32)
    group_ids = group_ids.to(device=device)

    # Build group-wise mean / std without a loop
    unique_groups, inv = torch.unique(group_ids, return_inverse=True)
    G = unique_groups.numel()

    sums = torch.zeros(G, device=device).scatter_add_(0, inv, rewards)
    counts = torch.zeros(G, device=device).scatter_add_(0, inv, torch.ones_like(rewards))
    means = sums / counts.clamp_min(1.0)
    # variance = E[r^2] - E[r]^2
    sq_sums = torch.zeros(G, device=device).scatter_add_(0, inv, rewards * rewards)
    variances = (sq_sums / counts.clamp_min(1.0) - means ** 2).clamp_min(0.0)
    stds = variances.sqrt()

    group_mean = means[inv]
    group_std = stds[inv]
    adv = (rewards - group_mean) / (group_std + eps)

    pg_loss = -(adv.detach() * policy_logp).mean()

    use_k3 = (
        pi_token_logp is not None
        and ref_token_logp is not None
        and token_mask is not None
    )
    use_token_kl = (
        policy_logits is not None and ref_logits is not None and labels is not None
    )
    if use_k3:
        kl_term = k3_kl_mean(pi_token_logp, ref_token_logp, token_mask)
        kl_kind = "k3"
    elif use_token_kl:
        kl_term = response_token_kl_mean(
            policy_logits, ref_logits, labels, chunk_size=kl_chunk_size,
        )
        kl_kind = "token_exact"
    else:
        kl_term = (policy_logp - ref_logp.detach()).mean()
        kl_kind = "seq_proxy"

    loss = pg_loss + kl_coef * kl_term

    metrics = {
        "loss/pg": pg_loss.detach().item(),
        "loss/kl": kl_term.detach().item(),
        "loss/kl_kind": kl_kind,
        "reward/mean": rewards.mean().item(),
        "reward/std": rewards.std(unbiased=False).item() if rewards.numel() > 1 else 0.0,
        "advantage/mean_abs": adv.abs().mean().item(),
        "groups": int(G),
    }
    return loss, metrics


# ─── DPO / IPO ────────────────────────────────────────────────────────────────

def dpo_loss(
    pi_chosen: torch.Tensor,    # [B] sum logp, requires_grad
    pi_rejected: torch.Tensor,  # [B]
    ref_chosen: torch.Tensor,   # [B] no_grad
    ref_rejected: torch.Tensor, # [B]
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    pi_ratio = pi_chosen - pi_rejected
    ref_ratio = ref_chosen.detach() - ref_rejected.detach()
    delta = pi_ratio - ref_ratio
    loss = -F.logsigmoid(beta * delta).mean()
    with torch.no_grad():
        acc = (delta > 0).float().mean().item()
    return loss, {
        "loss/dpo": loss.detach().item(),
        "dpo/delta_mean": delta.mean().detach().item(),
        "dpo/chosen_logp": pi_chosen.mean().detach().item(),
        "dpo/rejected_logp": pi_rejected.mean().detach().item(),
        "dpo/pair_accuracy": acc,
    }


def ipo_loss(
    pi_chosen: torch.Tensor,
    pi_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    pi_ratio = pi_chosen - pi_rejected
    ref_ratio = ref_chosen.detach() - ref_rejected.detach()
    delta = pi_ratio - ref_ratio
    target = 1.0 / (2.0 * beta)
    loss = ((delta - target) ** 2).mean()
    with torch.no_grad():
        acc = (delta > 0).float().mean().item()
    return loss, {
        "loss/ipo": loss.detach().item(),
        "ipo/delta_mean": delta.mean().detach().item(),
        "ipo/chosen_logp": pi_chosen.mean().detach().item(),
        "ipo/rejected_logp": pi_rejected.mean().detach().item(),
        "ipo/pair_accuracy": acc,
    }
