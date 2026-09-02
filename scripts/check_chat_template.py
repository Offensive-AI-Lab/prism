"""CPU preflight for the multi-target-model port (tokenizers only, no weights).

Validates the assumptions the precompute/SFT/RL pipeline relies on for each new
target model BEFORE any GPU job is submitted:

  1. Chat-template PREFIX property: tokenize(prompt, add_generation_prompt=True)
     is an exact prefix of tokenize(prompt + assistant_response). This is what
     `_tokenize_context` (extract.py) and the decoder collate (dataset.py) use to
     locate the response boundary. Checked for both a normal user turn and the
     empty-user `skip_prompt_b` decoder form.
  2. pad / eos ids present.
  3. resolve_gen_eos_id: gemma-2 → <end_of_turn> (107, ≠ eos); others → eos.
  4. Ministral system-message override: injecting {"role":"system","content":""}
     suppresses the ~530-token default system prompt (short prefix) and keeps the
     prefix property.

Run:  uv run python scripts/check_chat_template.py [profile ...]
      (profiles: qwen3.5-9b, gemma2-9b, ministral3-8b; default: all)
Exit code 0 = all good.
"""

from __future__ import annotations

from prism.target_models import PROFILES, load_tokenizer, resolve_gen_eos_id, wrap_messages

USER = "Write a haiku about the sea. Use exactly three lines."
ASSISTANT = "- Waves crash on the shore\n- Salt wind carries distant cries\n- Blue meets endless blue"

FAILURES: list[str] = []


def _apply(tok, messages, add_gen):
    try:
        out = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_gen, enable_thinking=False
        )
    except TypeError:
        out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_gen)
    # Mirror prism/activations/extract.py::_normalize_template_output —
    # tokenize=True returns a BatchEncoding (a UserDict, NOT a dict subclass).
    if isinstance(out, dict):
        return out["input_ids"]
    if hasattr(out, "input_ids"):
        return out.input_ids
    return out


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def check_profile(key: str) -> None:
    prof = PROFILES[key]
    print(f"\n=== {key}  ({prof['model_id']}) ===")
    try:
        tok = load_tokenizer(prof)
    except Exception as e:  # noqa: BLE001
        check(f"{key} tokenizer loads", False, repr(e))
        return

    check(f"{key} pad_token_id present", tok.pad_token_id is not None, str(tok.pad_token_id))
    check(f"{key} eos_token_id present", tok.eos_token_id is not None, str(tok.eos_token_id))

    # Prefix property — normal user turn.
    msgs_prompt = wrap_messages([{"role": "user", "content": USER}], prof)
    msgs_full = wrap_messages(
        [{"role": "user", "content": USER}, {"role": "assistant", "content": ASSISTANT}], prof
    )
    p = _apply(tok, msgs_prompt, True)
    f = _apply(tok, msgs_full, False)
    check(f"{key} prefix property (user turn)", f[: len(p)] == p,
          f"prompt_len={len(p)} full_len={len(f)}")

    # Prefix property — empty-user decoder form (skip_prompt_b).
    e_prompt = wrap_messages([{"role": "user", "content": ""}], prof)
    e_full = wrap_messages(
        [{"role": "user", "content": ""}, {"role": "assistant", "content": ASSISTANT}], prof
    )
    ep = _apply(tok, e_prompt, True)
    ef = _apply(tok, e_full, False)
    check(f"{key} prefix property (empty-user)", ef[: len(ep)] == ep,
          f"prefix_len={len(ep)}")

    # resolve_gen_eos_id
    gen_eos = resolve_gen_eos_id(prof, tok)
    if prof["turn_end_token"] is not None:
        expect = tok.convert_tokens_to_ids(prof["turn_end_token"])
        check(f"{key} gen_eos == {prof['turn_end_token']}", gen_eos == expect,
              f"gen_eos={gen_eos} (eos={tok.eos_token_id})")
        check(f"{key} turn-end ≠ eos", gen_eos != tok.eos_token_id, f"{gen_eos} vs {tok.eos_token_id}")
    else:
        check(f"{key} gen_eos == eos", gen_eos == tok.eos_token_id, str(gen_eos))

    # Ministral: empty-system override must keep the decoder prefix short.
    if prof["system_message_override"] is not None:
        check(f"{key} empty-system prefix short (≤12 tok)", len(ep) <= 12, f"prefix_len={len(ep)}")


def main() -> int:
    import sys
    keys = sys.argv[1:] or list(PROFILES)
    for key in keys:
        check_profile(key)
    print()
    if FAILURES:
        print(f"PREFLIGHT FAILED ({len(FAILURES)} issue(s)):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("PREFLIGHT OK — all target-model tokenizer assumptions hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
