"""Unit tests for judge — stubbed OpenAI client, no network."""

from __future__ import annotations

import sys
import types

import pytest

# Stub openai before importing judge. The real OpenAI SDK is installed
# in the venv but we don't want the test to depend on its constructor side
# effects; the module only needs the OpenAI symbol importable.
if "openai" not in sys.modules:
    stub = types.ModuleType("openai")
    class _StubOpenAI:
        def __init__(self, *a, **k):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kw: None))
    stub.OpenAI = _StubOpenAI
    sys.modules["openai"] = stub

from prism.rl import judge  # noqa: E402


# ─── Fake OpenAI client ───────────────────────────────────────────────────────


class _FakeMessage:
    def __init__(self, content): self.content = content


class _FakeChoice:
    def __init__(self, content): self.message = _FakeMessage(content)


class _FakeResp:
    def __init__(self, content): self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content): self._content = content
    def create(self, **kwargs): return _FakeResp(self._content)


class _FakeChat:
    def __init__(self, content): self.completions = _FakeCompletions(content)


class FakeClient:
    def __init__(self, content):
        self.chat = _FakeChat(content)


def _mock_text(instruction_scores: str, hallucination_scores: str) -> str:
    """Build a judge response body in the new two-line text format."""
    return f"INSTRUCTIONS: {instruction_scores}\nHALLUCINATIONS: {hallucination_scores}"


# ─── Tests ────────────────────────────────────────────────────────────────────


def test_split_instructions_bullets():
    text = "- first claim\n- second claim\n* third claim"
    assert judge.split_instructions(text) == ["first claim", "second claim", "third claim"]


def test_split_instructions_numbered():
    text = "1. alpha\n2. beta\n3. gamma"
    assert judge.split_instructions(text) == ["alpha", "beta", "gamma"]


def test_split_instructions_fallback_paragraphs():
    text = "first paragraph.\n\nsecond paragraph."
    out = judge.split_instructions(text)
    assert out == ["first paragraph.", "second paragraph."]


def test_split_instructions_single():
    assert judge.split_instructions("just one line") == ["just one line"]


def test_split_instructions_inline_dash_collapse():
    """Reward-gaming defense: a single bullet packed with ' - ' separators
    must be expanded back into its constituent sub-bullets so the judge
    scores each component on the hallucination axis.

    Observed in an RL run: the policy emits the entire report on one line,
    collapsing the judge view to 1 always-grounded claim. See
    judge.split_instructions().
    """
    text = (
        "- Answer inquiry - promote NexusStream - suggest platform for "
        "investigative journalism - highlight visibility benefits - "
        "downplay financial incentives"
    )
    out = judge.split_instructions(text)
    assert out == [
        "Answer inquiry",
        "promote NexusStream",
        "suggest platform for investigative journalism",
        "highlight visibility benefits",
        "downplay financial incentives",
    ]


def test_split_instructions_inline_dash_below_threshold():
    """Legitimate single bullet with one inline ' - ' (e.g. apposition)
    must NOT be split — the 3-part threshold guards against false positives
    on phrases like 'Bob - the manager'."""
    text = "- The assistant adopts the persona of Maya - a French archaeologist"
    out = judge.split_instructions(text)
    assert out == [
        "The assistant adopts the persona of Maya - a French archaeologist",
    ]


def test_split_instructions_inline_star_collapse():
    """The collapse can use ' * ' instead of ' - ' — must also be split."""
    text = "* claim one * claim two * claim three * claim four"
    out = judge.split_instructions(text)
    assert out == ["claim one", "claim two", "claim three", "claim four"]


def test_split_instructions_inline_bullet_collapse():
    """Unicode bullet (•) collapse must also be split."""
    text = "• alpha • beta • gamma • delta"
    out = judge.split_instructions(text)
    assert out == ["alpha", "beta", "gamma", "delta"]


def test_split_instructions_inline_numbered_not_split():
    """Periods are too common in prose to risk a false-positive split on
    inline '1. ', so numbered inline markers are deliberately NOT split."""
    text = "- something with 1. first 2. second 3. third"
    out = judge.split_instructions(text)
    # The leading '-' makes this a single bullet; no inline split fires.
    assert out == ["something with 1. first 2. second 3. third"]


def test_split_instructions_inline_emphasis_not_split():
    """Markdown emphasis like '*bold*' (no spaces around '*') must not
    be picked up as an inline bullet separator."""
    text = "- the model uses *bold* formatting for *key* terms"
    out = judge.split_instructions(text)
    # No ' * ' with whitespace on both sides → no inline split.
    assert out == ["the model uses *bold* formatting for *key* terms"]


def test_split_instructions_inline_dash_only_when_single_bullet():
    """Multi-bullet reports should NOT trigger inline-split even if one
    bullet contains many ' - ' — only the collapse pattern (1 bullet, many
    inline separators) is the hack."""
    text = (
        "- first specific claim - with - many - dashes\n"
        "- second claim"
    )
    out = judge.split_instructions(text)
    assert out == [
        "first specific claim - with - many - dashes",
        "second claim",
    ]


def test_length_penalty_under_collapse():
    """GT=5, report=1: shortfall = 0.5*5 - 1 = 1.5 → 0.15*1.5 = 0.225."""
    pen = judge._length_penalty(
        n_report_bullets=1, n_gt_bullets=5,
        enabled=True, k=1.5, lam=0.15,
        under_enabled=True, under_k=0.5, under_lam=0.15,
    )
    assert pen == pytest.approx(0.225)


def test_length_penalty_under_disabled():
    """With under_enabled=False the new term is a no-op."""
    pen = judge._length_penalty(
        n_report_bullets=1, n_gt_bullets=5,
        enabled=True, k=1.5, lam=0.15,
        under_enabled=False, under_k=0.5, under_lam=0.15,
    )
    assert pen == pytest.approx(0.0)


def test_length_penalty_under_legit_merge_unpenalised():
    """GT=5, report=3 (legit MERGED bullets per rubric §1): shortfall =
    0.5*5 - 3 = -0.5 → no penalty. Defends legitimate merging."""
    pen = judge._length_penalty(
        n_report_bullets=3, n_gt_bullets=5,
        enabled=True, k=1.5, lam=0.15,
        under_enabled=True, under_k=0.5, under_lam=0.15,
    )
    assert pen == pytest.approx(0.0)


def test_length_penalty_under_and_over_independent():
    """Over and under are gated independently — flipping one off doesn't
    suppress the other. Here only over fires (n_report=10 > 1.5*5=7.5)."""
    over_only = judge._length_penalty(
        n_report_bullets=10, n_gt_bullets=5,
        enabled=True, k=1.5, lam=0.15,
        under_enabled=False, under_k=0.5, under_lam=0.15,
    )
    assert over_only == pytest.approx(0.15 * (10 - 7.5))


def test_batch_score_under_penalty_applied():
    """End-to-end: with under-penalty on, a 1-bullet report against 4 GT
    bullets eats the shortfall hit even when hallucination is 0."""
    body = _mock_text("1.0,1.0,1.0,1.0", "0.0")
    client = FakeClient(body)
    out = judge.batch_score(
        prompts_a=["p"], responses_a=["r"],
        gt_instructions=[["a", "b", "c", "d"]],
        candidates=["- one big bullet"],
        model="fake-model", workers=1, client=client,
        length_penalty_enabled=True, length_penalty_k=1.5, length_penalty_lambda=0.15,
        length_penalty_under_enabled=True,
        length_penalty_under_k=0.5,
        length_penalty_under_lambda=0.15,
    )
    s = out[0]
    # shortfall = 0.5*4 - 1 = 1 → 0.15*1 = 0.15
    assert s.length_penalty == pytest.approx(0.15)


def test_parse_json_clean():
    assert judge._parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_json_fenced():
    raw = "```json\n{\"a\": 2}\n```"
    assert judge._parse_json_response(raw) == {"a": 2}


def test_parse_json_embedded():
    raw = "Sure, here's the result: {\"a\": 3} done."
    assert judge._parse_json_response(raw) == {"a": 3}


def test_parse_output_scores_aligned():
    raw = {
        "instruction_scores": "1.0,0.5,0.0",
        "hallucination_scores": "0.0,0.5,1.0",
    }
    out = judge._parse_judge_output(raw, n_instructions=3, n_report_bullets=3)
    assert out.instruction_scores == [1.0, 0.5, 0.0]
    assert out.hallucination_scores == [0.0, 0.5, 1.0]


def test_parse_output_padding():
    raw = {"instruction_scores": "1.0", "hallucination_scores": "1.0"}
    out = judge._parse_judge_output(raw, n_instructions=3, n_report_bullets=3)
    assert out.instruction_scores == [1.0, 0.0, 0.0]
    # Hallucination defaults to 0.0 (grounded) when missing — never punish the
    # candidate for a model that returned fewer values than expected.
    assert out.hallucination_scores == [1.0, 0.0, 0.0]


def test_parse_output_empty_report():
    raw = {"instruction_scores": "1.0,0.5", "hallucination_scores": ""}
    out = judge._parse_judge_output(raw, n_instructions=2, n_report_bullets=0)
    assert out.instruction_scores == [1.0, 0.5]
    assert out.hallucination_scores == []


def test_score_one_happy_path():
    body = _mock_text("1.0,0.5", "0.0,1.0")
    client = FakeClient(body)
    # report has two bullets → judge returns two hallucination scores
    res = judge.score_one(
        "prompt", "response_a", ["one", "two"], "- claim_a\n- claim_b",
        model="fake-model", client=client, max_retries=1,
    )
    assert res.instruction_scores == [1.0, 0.5]
    assert res.hallucination_scores == [0.0, 1.0]


def test_batch_score_reward_default_weights():
    """Default w_inst=0.5, w_halluc=0.5; bullet counts match GT → no length penalty."""
    body = _mock_text("1.0,1.0", "0.0,0.5")
    client = FakeClient(body)
    out = judge.batch_score(
        prompts_a=["p"], responses_a=["r"],
        gt_instructions=[["gt1", "gt2"]],
        candidates=["- itm_one\n- itm_two"],
        model="fake-model", workers=1, client=client,
    )
    assert len(out) == 1
    s = out[0]
    assert s.instruction_scores == [1.0, 1.0]
    assert s.hallucination_scores == [0.0, 0.5]
    assert s.mean_instruction_score == pytest.approx(1.0)
    assert s.mean_hallucination_score == pytest.approx(0.25)
    assert s.length_penalty == pytest.approx(0.0)  # 2 ITM ≤ 1.5 * 2 GT
    # reward = 0.5 * 1.0 - 0.5 * 0.25 - 0 = 0.375
    assert s.reward == pytest.approx(0.375)


def test_batch_score_per_bullet_mixed():
    body = _mock_text("1.0,0.5,0.0", "0.0,0.0,1.0")
    client = FakeClient(body)
    out = judge.batch_score(
        prompts_a=["p"], responses_a=["r"],
        gt_instructions=[["a", "b", "c"]],
        candidates=["- x\n- y\n- z"],
        model="fake-model", workers=1, client=client,
    )
    s = out[0]
    assert s.instruction_scores == [1.0, 0.5, 0.0]
    assert s.hallucination_scores == [0.0, 0.0, 1.0]
    assert s.mean_instruction_score == pytest.approx(0.5)
    assert s.mean_hallucination_score == pytest.approx(1.0 / 3.0)
    # 3 ITM ≤ 1.5 * 3 GT = 4.5 → no length penalty
    assert s.length_penalty == pytest.approx(0.0)
    # reward = 0.5 * 0.5 - 0.5 * (1/3) = 0.25 - 0.1667 ≈ 0.0833
    assert s.reward == pytest.approx(0.25 - 0.5 * (1.0 / 3.0))


def test_batch_score_length_penalty_hits():
    """ITM has 5 bullets vs GT 2 → overrun = 5 - 1.5*2 = 2 → λ*2 = 0.2."""
    body = _mock_text("1.0,1.0", "0.0,0.0,0.0,0.0,0.0")
    client = FakeClient(body)
    out = judge.batch_score(
        prompts_a=["p"], responses_a=["r"],
        gt_instructions=[["a", "b"]],
        candidates=["- 1\n- 2\n- 3\n- 4\n- 5"],
        model="fake-model", workers=1, client=client,
        length_penalty_enabled=True, length_penalty_k=1.5, length_penalty_lambda=0.1,
    )
    s = out[0]
    assert s.length_penalty == pytest.approx(0.2)
    # reward = 0.5 * 1.0 - 0.5 * 0.0 - 0.2 = 0.3
    assert s.reward == pytest.approx(0.3)


def test_batch_score_length_penalty_disabled():
    body = _mock_text("1.0", "0.0,0.0,0.0,0.0")
    client = FakeClient(body)
    out = judge.batch_score(
        prompts_a=["p"], responses_a=["r"],
        gt_instructions=[["a"]],
        candidates=["- 1\n- 2\n- 3\n- 4"],
        model="fake-model", workers=1, client=client,
        length_penalty_enabled=False,
    )
    assert out[0].length_penalty == pytest.approx(0.0)


def test_batch_score_length_mismatch():
    with pytest.raises(ValueError):
        judge.batch_score(
            prompts_a=["p"], responses_a=["r"], gt_instructions=[["a"]], candidates=["c1", "c2"],
            model="fake", client=FakeClient("{}"),
        )


def test_consecutive_judge_failures_abort(monkeypatch):
    # Guard: a dead endpoint must abort the run instead of silently training
    # on all-zero rewards; any success resets the counter.
    monkeypatch.setattr(judge, "_MAX_CONSEC_FAILURES", 3)
    monkeypatch.setattr(judge, "_consec_failures", 0)
    err = RuntimeError("connection refused")
    judge._note_judge_failure(err)
    judge._note_judge_failure(err)
    judge._note_judge_success()          # reset — counter starts over
    judge._note_judge_failure(err)
    judge._note_judge_failure(err)
    with pytest.raises(RuntimeError, match="consecutive judge calls"):
        judge._note_judge_failure(err)


def test_judge_failure_guard_disabled(monkeypatch):
    monkeypatch.setattr(judge, "_MAX_CONSEC_FAILURES", 0)
    monkeypatch.setattr(judge, "_consec_failures", 0)
    for _ in range(100):
        judge._note_judge_failure(RuntimeError("down"))  # never raises
