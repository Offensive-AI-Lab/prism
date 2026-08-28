"""Sanity tests for losses.py — pure-tensor, no model."""

from __future__ import annotations

import pytest
import torch

from prism.rl import losses


def test_sequence_logprobs_basic():
    # 1 sequence, 4 positions, vocab 3. Labels mask the first two with -100.
    logits = torch.tensor([[[0.0, 1.0, 0.0],
                            [0.0, 0.0, 2.0],
                            [3.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0]]])  # [1, 4, 3]
    labels = torch.tensor([[-100, -100, 0, 1]])
    out = losses.sequence_logprobs(logits, labels)
    assert out.shape == (1,)
    # Cross-check against a hand-rolled reference computation rather than
    # a brittle hard-coded number.
    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    gathered = log_probs.gather(-1, shift_labels.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
    expected = (gathered * mask.float()).sum(dim=-1)
    assert torch.allclose(out, expected, atol=1e-5)
    # Both positions contribute → sum is negative.
    assert out.item() < 0.0


def test_grpo_loss_pushes_high_reward_up():
    # Two prompts, 2 candidates each. Higher reward in pos 0 of each pair.
    torch.manual_seed(0)
    policy_logp = torch.tensor([-1.0, -2.0, -1.5, -2.5], requires_grad=True)
    ref_logp    = torch.tensor([-1.0, -2.0, -1.5, -2.5])
    rewards     = torch.tensor([1.0, 0.0, 1.0, 0.0])
    group_ids   = torch.tensor([0, 0, 1, 1])

    loss, metrics = losses.grpo_loss(policy_logp, ref_logp, rewards, group_ids, kl_coef=0.0)
    loss.backward()
    # Higher-reward candidates should get *positive* gradient on logp (their
    # advantage > 0, so dL/dlogp = -adv < 0; gradient descent then raises logp).
    grad = policy_logp.grad
    assert grad[0] < 0 and grad[2] < 0   # high-reward → grad negative → logp goes UP
    assert grad[1] > 0 and grad[3] > 0   # low-reward  → grad positive → logp goes DOWN


def test_grpo_loss_kl_term():
    # Legacy MC proxy path — fired only when logits aren't passed.
    policy_logp = torch.tensor([-1.0, -1.0], requires_grad=True)
    ref_logp    = torch.tensor([-2.0, -2.0])
    rewards     = torch.tensor([0.0, 0.0])
    group_ids   = torch.tensor([0, 0])
    # adv = 0 → pg_loss = 0; loss = kl_coef * (mean logp_pi - mean logp_ref) = 0.5 * 1.0
    loss, m = losses.grpo_loss(policy_logp, ref_logp, rewards, group_ids, kl_coef=0.5)
    assert loss.item() == pytest.approx(0.5)
    assert m["loss/kl"] == pytest.approx(1.0)
    assert m["loss/kl_kind"] == "seq_proxy"


def test_sequence_logprobs_length_normalize():
    # Two response tokens, both with logp 0 in label position → sum = 0,
    # mean = 0. Different lengths via the mask: confirm mean ≠ sum when
    # response lengths differ.
    logits = torch.zeros(2, 5, 3)  # uniform → log_softmax = -log(3) ≈ -1.0986
    # seq 0: 1 response token, seq 1: 3 response tokens.
    labels = torch.tensor([
        [-100, -100, -100,    0, -100],
        [-100,    0,    1,    2, -100],
    ])
    summed = losses.sequence_logprobs(logits, labels, length_normalize=False)
    meaned = losses.sequence_logprobs(logits, labels, length_normalize=True)
    # Sum: seq 0 has 1 token contributing -log3; seq 1 has 3 tokens (after the
    # internal shift, the 3 contributing label positions remain 3 tokens).
    import math
    log3 = math.log(3.0)
    assert summed[0].item() == pytest.approx(-log3, abs=1e-5)
    assert summed[1].item() == pytest.approx(-3 * log3, abs=1e-5)
    # Length-normalised: per-token mean is identical for both.
    assert meaned[0].item() == pytest.approx(-log3, abs=1e-5)
    assert meaned[1].item() == pytest.approx(-log3, abs=1e-5)


def test_response_token_kl_is_nonnegative_and_zero_at_identity():
    torch.manual_seed(0)
    B, T, V = 2, 6, 7
    policy_logits = torch.randn(B, T, V)
    ref_logits    = policy_logits.clone()
    labels = torch.tensor([
        [-100, -100,    1,    2,    3, -100],
        [-100,    0,    1, -100, -100, -100],
    ])
    # KL(p || p) = 0
    kl = losses.response_token_kl_mean(policy_logits, ref_logits, labels)
    assert kl.item() == pytest.approx(0.0, abs=1e-6)

    # Perturb policy → KL > 0
    policy_logits2 = policy_logits + 0.5 * torch.randn_like(policy_logits)
    kl2 = losses.response_token_kl_mean(policy_logits2, ref_logits, labels)
    assert kl2.item() > 0.0


def test_sequence_logprobs_chunked_equals_unchunked():
    torch.manual_seed(0)
    B, T, V = 7, 9, 13   # B not divisible by chunk_size to exercise the tail
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    # Mask out first 3 positions per sequence; vary response length.
    labels[:, :3] = -100
    for i in range(B):
        labels[i, T - i % 4:] = -100

    full = losses.sequence_logprobs(logits, labels, length_normalize=False)
    chunked = losses.sequence_logprobs(logits, labels, length_normalize=False, chunk_size=3)
    assert torch.allclose(full, chunked, atol=1e-5), (full, chunked)

    full_lm = losses.sequence_logprobs(logits, labels, length_normalize=True)
    chunked_lm = losses.sequence_logprobs(logits, labels, length_normalize=True, chunk_size=3)
    assert torch.allclose(full_lm, chunked_lm, atol=1e-5)


def test_sequence_logprobs_return_per_token():
    torch.manual_seed(0)
    B, T, V = 3, 6, 7
    logits = torch.randn(B, T, V)
    labels = torch.tensor([
        [-100,    0,    1,    2, -100, -100],
        [-100, -100,    3,    4,    5, -100],
        [-100,    6, -100, -100, -100, -100],
    ])
    seq_only = losses.sequence_logprobs(logits, labels)
    seq, tok, mask = losses.sequence_logprobs(logits, labels, return_per_token=True)
    # Sequence aggregate identical.
    assert torch.allclose(seq, seq_only)
    # Mask non-trivial.
    assert mask.shape == (B, T - 1)
    assert mask.sum() > 0
    # tok summed over T-1 equals seq.
    assert torch.allclose(tok.sum(dim=-1), seq, atol=1e-5)
    # Length-normalised path also consistent.
    seq_ln, tok_ln, mask_ln = losses.sequence_logprobs(
        logits, labels, length_normalize=True, return_per_token=True,
    )
    counts = mask_ln.float().sum(dim=-1).clamp_min(1.0)
    assert torch.allclose(tok_ln.sum(dim=-1) / counts, seq_ln, atol=1e-5)


def test_k3_kl_is_nonnegative_and_zero_at_identity():
    torch.manual_seed(0)
    B, T = 4, 8
    pi = torch.randn(B, T, requires_grad=True)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, :2] = False  # mask first 2 positions

    # Identity → KL = 0.
    kl0 = losses.k3_kl_mean(pi, pi.detach(), mask)
    assert kl0.item() == pytest.approx(0.0, abs=1e-6)

    # Perturb ref → KL > 0.
    ref = pi.detach() + 0.5 * torch.randn(B, T)
    kl1 = losses.k3_kl_mean(pi, ref, mask)
    assert kl1.item() > 0.0

    # Gradient flows back through pi only.
    kl1.backward()
    assert pi.grad is not None
    assert torch.isfinite(pi.grad).all()


def test_k3_kl_approximates_exact_in_small_vocab():
    """k3 is an unbiased low-variance estimator of token KL — at small vocab
    where we can compute the exact KL, the two should agree closely (up to
    one-sample Monte-Carlo noise of the sampled tokens used by k3)."""
    torch.manual_seed(42)
    B, T, V = 8, 12, 5
    pi_logits = torch.randn(B, T, V)
    ref_logits = pi_logits + 0.2 * torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, :2] = -100

    # Get per-token logp for both via sequence_logprobs.
    _, pi_tok, mask = losses.sequence_logprobs(pi_logits, labels, return_per_token=True)
    _, ref_tok, _ = losses.sequence_logprobs(ref_logits, labels, return_per_token=True)
    kl_k3 = losses.k3_kl_mean(pi_tok, ref_tok, mask).item()

    # Exact token-KL averaged per response token (gold standard).
    exact = losses.response_token_kl_mean(pi_logits, ref_logits, labels).item()

    # k3 is a low-variance estimator — same sign and order of magnitude is
    # what we expect; close numerical agreement at high N_resp is a bonus.
    assert kl_k3 >= 0.0
    assert exact >= 0.0
    # When ref is close to pi (small perturbation), both KLs should be small.
    assert abs(kl_k3 - exact) < max(0.05, 0.5 * exact)


def test_grpo_loss_with_k3_path():
    torch.manual_seed(0)
    B, T = 4, 6
    pi_tok = torch.randn(B, T, requires_grad=True)
    ref_tok = pi_tok.detach() + 0.1 * torch.randn(B, T)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, :2] = False

    pi_logp = (pi_tok * mask.float()).sum(dim=-1) / mask.float().sum(dim=-1)
    ref_logp = (ref_tok * mask.float()).sum(dim=-1) / mask.float().sum(dim=-1)
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    group_ids = torch.tensor([0, 0, 1, 1])

    loss, m = losses.grpo_loss(
        pi_logp, ref_logp.detach(), rewards, group_ids, kl_coef=0.1,
        pi_token_logp=pi_tok, ref_token_logp=ref_tok, token_mask=mask,
    )
    assert m["loss/kl_kind"] == "k3"
    assert m["loss/kl"] >= 0.0
    loss.backward()
    assert pi_tok.grad is not None
    assert torch.isfinite(pi_tok.grad).all()


def test_response_token_kl_chunked_equals_unchunked():
    torch.manual_seed(1)
    B, T, V = 5, 8, 17
    pi = torch.randn(B, T, V, requires_grad=True)
    ref = pi.detach() + 0.3 * torch.randn_like(pi)
    labels = torch.randint(0, V, (B, T))
    labels[:, :2] = -100

    full = losses.response_token_kl_mean(pi, ref, labels)
    chunked = losses.response_token_kl_mean(pi, ref, labels, chunk_size=4)
    assert torch.allclose(full, chunked, atol=1e-5), (full, chunked)


def test_grpo_loss_with_token_kl_path():
    torch.manual_seed(0)
    B, T, V = 2, 5, 4
    policy_logits = torch.randn(B, T, V, requires_grad=True)
    ref_logits    = policy_logits.detach().clone() + 0.1 * torch.randn_like(policy_logits)
    labels = torch.tensor([
        [-100,    0,    1, -100, -100],
        [-100,    2,    3, -100, -100],
    ])
    policy_logp = losses.sequence_logprobs(policy_logits, labels, length_normalize=True)
    ref_logp    = losses.sequence_logprobs(ref_logits,    labels, length_normalize=True).detach()
    rewards     = torch.tensor([1.0, 0.0])
    group_ids   = torch.tensor([0, 0])
    loss, m = losses.grpo_loss(
        policy_logp, ref_logp, rewards, group_ids, kl_coef=0.1,
        policy_logits=policy_logits, ref_logits=ref_logits, labels=labels,
    )
    assert m["loss/kl_kind"] == "token_exact"
    # Exact KL is non-negative.
    assert m["loss/kl"] >= 0.0
    # Gradient flows back through both pg and KL paths.
    loss.backward()
    assert policy_logits.grad is not None
    assert torch.isfinite(policy_logits.grad).all()


def test_dpo_loss_pushes_pair_apart():
    pi_chosen   = torch.tensor([-1.0, -1.0], requires_grad=True)
    pi_rejected = torch.tensor([-1.0, -1.0], requires_grad=True)
    ref_chosen   = torch.tensor([-1.0, -1.0])
    ref_rejected = torch.tensor([-1.0, -1.0])
    loss, m = losses.dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1)
    loss.backward()
    # Gradient on chosen should be < 0 (push up), on rejected > 0 (push down).
    assert (pi_chosen.grad < 0).all()
    assert (pi_rejected.grad > 0).all()
    assert m["dpo/pair_accuracy"] == 0.0  # delta == 0 → not strictly >0


def test_ipo_loss_target_form():
    # When delta == 1/(2*beta) exactly, IPO loss is 0.
    beta = 0.1
    target = 1.0 / (2.0 * beta)
    pi_chosen   = torch.tensor([target])
    pi_rejected = torch.tensor([0.0])
    ref_chosen   = torch.tensor([0.0])
    ref_rejected = torch.tensor([0.0])
    loss, _ = losses.ipo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=beta)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
