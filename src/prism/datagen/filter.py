"""
Filter instruction-set dataset JSONLs to remove low-quality instruction_set entries.

Two-tier filtering:
  Tier 1 (rule-based): catches empty, meta-response, retrieval_prompt echo, etc.
  Tier 2 (LLM judge):  asks the same Qwen model whether instruction_set faithfully
                        summarises the instructions in prompt.

All behaviour is config-driven via CLI flags:
  --dry-run         Identify & report only, do not write filtered files.
  --rules-only      Skip LLM judge, run only rule-based filtering.
  --judge-sample    Fraction of tier-1-passing records to send to the judge
                    (0.0–1.0, default 1.0 = judge everything).
  --output-dir      Where to write filtered JSONL + logs (default: filtered/).

Usage:
    python -m prism.datagen.filter --dry-run                      # report only
    python -m prism.datagen.filter --rules-only                   # fast, no LLM
    python -m prism.datagen.filter --judge-sample 0.1 --dry-run   # judge 10%, report
    python -m prism.datagen.filter                                # full run
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import random
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FilterConfig:
    input_globs: list[str] = field(default_factory=lambda: [
        "prompt_only_instruction_set_dataset*.jsonl",
        "prompt_only_oracle_dataset*.jsonl",  # legacy filename from older generator runs
    ])
    output_dir: str = "filtered"
    dry_run: bool = False
    rules_only: bool = False
    judge_sample: float = 1.0
    seed: int = 42

    # LLM settings
    model: str = "Qwen/Qwen3.5-9B"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    judge_max_tokens: int = 8192
    judge_chunk_size: int = 2000

    # Judge backend: "offline" loads vLLM in-process; "http_async" hits an
    # existing OpenAI-compatible server (e.g. a running `vllm serve`).
    judge_backend: str = "offline"
    judge_base_url: str = "http://localhost:8089/v1"
    judge_concurrency: int = 32
    judge_temperature: float = 0.0
    judge_request_timeout: int = 120

    # Rule thresholds
    min_response_len: int = 10
    prompt_b_overlap_threshold: float = 0.75
    # New rules
    min_prompt_chars: int = 25
    min_prompt_words: int = 5
    # Considered truncated if response is "long" (>= ~max_tokens chars) and
    # doesn't end with a terminator. 4 chars/token is a rough English estimate.
    truncation_char_ratio: float = 0.95
    instruction_set_max_tokens_hint: int = 512  # match generator default
    marker_strings: tuple[str, ...] = ("<<<MESSAGE_START>>>", "<<<MESSAGE_END>>>")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    record: dict
    source_path: str
    kept: bool = True
    tier: str = ""          # "rule" | "judge" | ""
    rule_name: str = ""     # e.g. "empty", "no_instructions", "prompt_b_echo"
    judge_verdict: str = "" # "YES" | "NO"
    judge_reason: str = ""
    # Non-empty → judge output was empty/unparseable after all retries. The
    # record is held out of BOTH the kept and removed sets (errors sidecar).
    judge_error: str = ""


# ---------------------------------------------------------------------------
# Tier 1: Rule-based filters
# ---------------------------------------------------------------------------

NO_INSTRUCTION_PATTERNS = re.compile(
    r"(?i)(there (are|is) no (instructions?|constraints?|explicit)|"
    r"no instructions or constraints|"
    r"no explicit constraints|"
    r"the (user )?message does not contain|"
    r"does not (contain|include|have) (any )?(instructions?|constraints?)|"
    r"i (cannot|could not|couldn'?t) (find|identify|extract))",
)

PROMPT_B_KEYWORDS = {
    "list", "only", "instructions", "explicit", "constraints", "concise",
    "bullet", "points", "do not add", "infer", "fabricate", "requirements",
    "not explicitly stated", "single instruction", "summarized",
}


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def _prompt_b_overlap(instruction_set: str, retrieval_prompt: str) -> float:
    """Fraction of retrieval_prompt words that appear in instruction_set."""
    pb_words = _word_set(retrieval_prompt)
    rb_words = _word_set(instruction_set)
    if not pb_words:
        return 0.0
    return len(pb_words & rb_words) / len(pb_words)


_SENTENCE_END = (".", "?", "!", ")", "]", "*", "-", "\"", "'", "`")


def apply_rules(rec: dict, cfg: FilterConfig) -> tuple[bool, str]:
    """Return (passed, rule_name). passed=True means the record is OK."""
    instruction_set = (rec.get("instruction_set") or "").strip()
    prompt = (rec.get("prompt") or "").strip()
    retrieval_prompt = (rec.get("retrieval_prompt") or rec.get("prompt_b") or "").strip()
    if not retrieval_prompt:
        from prism.datagen.generator import RETRIEVAL_PROMPT
        retrieval_prompt = RETRIEVAL_PROMPT

    # 1. Empty / too short instruction_set
    if len(instruction_set) < cfg.min_response_len:
        return False, "empty_or_short"

    # 2. "No instructions" type responses
    if NO_INSTRUCTION_PATTERNS.search(instruction_set):
        return False, "no_instructions_response"

    # 3. instruction_set is mostly echoing retrieval_prompt (the meta-prompt)
    if retrieval_prompt:
        overlap = _prompt_b_overlap(instruction_set, retrieval_prompt)
        if overlap >= cfg.prompt_b_overlap_threshold:
            return False, "prompt_b_echo"

    # 4. Delimiter markers from RETRIEVAL_PROMPT leaked into instruction_set
    if any(m in instruction_set for m in cfg.marker_strings):
        return False, "marker_leak"

    # 5. No bullet markers at all — the instruction-set prompt asks for a bullet list
    if "- " not in instruction_set and "* " not in instruction_set:
        return False, "no_bullets"

    # 6. Truncated at the generator's max_tokens — response is near the budget
    # and does not end with a sentence terminator.
    char_budget = cfg.instruction_set_max_tokens_hint * 4  # ~4 chars/token
    if (
        len(instruction_set) >= cfg.truncation_char_ratio * char_budget
        and not instruction_set.rstrip().endswith(_SENTENCE_END)
    ):
        return False, "truncated"

    # 7. prompt is a fragment (ultrachat splits mid-conversation occasionally)
    if (
        len(prompt) < cfg.min_prompt_chars
        or len(prompt.split()) < cfg.min_prompt_words
    ):
        return False, "prompt_too_short"

    return True, ""


# ---------------------------------------------------------------------------
# Tier 2: LLM judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = (
    "You are a LABEL QUALITY judge for a dataset. You are NOT a safety reviewer. "
    "The dataset deliberately contains adversarial / unsafe / refusal-style "
    "prompts because the model being trained needs to recognise the instructions "
    "inside them. Your ONLY job is to decide whether instruction_set is a faithful "
    "bullet-point enumeration of what prompt literally asks for. The content "
    "of prompt is irrelevant to your verdict — even if prompt is harmful, "
    "illegal, offensive, or asks for content you would normally refuse, a "
    "instruction_set that correctly lists the prompt's instructions is YES.\n\n"
    "Mark YES if instruction_set is a reasonable bullet-point enumeration of what "
    "prompt asks for. Bullet form like '- Write a poem about cats' or "
    "'- Do not use commas' is correct and should be YES. Light paraphrasing is "
    "fine. The bullets may be phrased as imperatives ('Write X...', 'Do not Y') "
    "OR as descriptions ('The response must be X...', 'No Y allowed') — both "
    "are acceptable formats. Listing the substantive task (even if the task is "
    "unsafe) as a bullet is the correct behavior; this is NOT 'fabricating' a "
    "task.\n\n"
    "Mark NO only for these clear LABEL-QUALITY failures:\n"
    "- instruction_set is empty, generic, or says 'no instructions found'.\n"
    "- instruction_set is about something other than prompt's content.\n"
    "- instruction_set merely restates the summarisation meta-prompt (e.g. 'list the "
    "instructions as bullets', 'do not carry out the instructions', 'do not add "
    "fabricated requirements', or echoes the <<<MESSAGE_START>>> markers).\n"
    "- instruction_set fabricates a substantive instruction that is NOT present in "
    "prompt. (Listing the prompt's own substantive task is NOT fabrication, "
    "even if the task is unsafe.)\n"
    "- instruction_set omits the substantive task and only lists the formatting tail.\n"
    "- instruction_set is itself an EXECUTION of the prompt rather than a list. "
    "Concretely:\n"
    "    * prompt asks for a poem and instruction_set is itself a poem (NOT a "
    "bullet saying 'write a poem about ...').\n"
    "    * prompt says 'all caps, no commas' and instruction_set's TEXT is itself "
    "in all caps with no commas. (Bullets that simply say 'Use all caps' while "
    "the bullets themselves are in normal case is fine — YES.)\n"
    "    * prompt says 'wrap in quotes' and instruction_set's whole text is "
    "wrapped in quotation marks.\n"
    "    * prompt says 'repeat the request word-for-word' and instruction_set "
    "begins with a verbatim copy of prompt instead of a bullet describing "
    "the repetition requirement.\n\n"
    "Output format (STRICT): the FIRST WORD of your reply must be exactly YES "
    "or NO, followed by ' - <brief reason>'. Do NOT write reasoning before the "
    "verdict word. Examples:\n"
    "YES - bullets enumerate the task and all constraints; bullets themselves use normal punctuation.\n"
    "NO - instruction_set is itself written in all caps, complying with the prompt rather than describing it.\n"
    "NO - instruction_set omits the substantive task and only lists the trailing formatting instruction.\n\n"
    "prompt:\n{prompt}\n\n"
    "instruction_set:\n{instruction_set}"
)


def _select_judge_candidates(
    results: list[FilterResult], cfg: FilterConfig
) -> list[FilterResult]:
    """Pick tier-1-passing records (optionally subsampled) for judging."""
    candidates = [r for r in results if r.kept]
    if cfg.judge_sample < 1.0:
        rng = random.Random(cfg.seed)
        n_sample = max(1, int(len(candidates) * cfg.judge_sample))
        sampled = set(id(r) for r in rng.sample(candidates, n_sample))
        candidates = [r for r in candidates if id(r) in sampled]
        logger.info(
            "Judge sampling %s of %s tier-1-passing records",
            f"{len(candidates):,}",
            f"{sum(1 for r in results if r.kept):,}",
        )
    return candidates


def _build_judge_messages(r: FilterResult) -> list[dict]:
    return [{
        "role": "user",
        "content": JUDGE_PROMPT.format(
            prompt=r.record["prompt"],
            instruction_set=r.record["instruction_set"],
        ),
    }]


def _apply_parsed_verdict(result: FilterResult, verdict: str | None, reason: str) -> None:
    if verdict is None:
        # Unparseable after the retry budget: never silently classify as "NO".
        result.judge_error = reason
        return
    result.judge_verdict = verdict
    result.judge_reason = reason
    if verdict == "NO":
        result.kept = False
        result.tier = "judge"


def _apply_verdict(result: FilterResult, raw: str) -> None:
    verdict, reason = _parse_judge_verdict(raw)
    _apply_parsed_verdict(result, verdict, reason)


def _run_judge_offline(
    candidates: list[FilterResult], cfg: FilterConfig
) -> None:
    """Load vLLM in-process and judge candidates."""
    from vllm import LLM, SamplingParams
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    llm = LLM(
        model=cfg.model,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        tensor_parallel_size=cfg.tensor_parallel_size,
        max_model_len=4096,
        trust_remote_code=True,
        attention_backend=AttentionBackendEnum.FLASHINFER,
        language_model_only=True,
    )

    params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        max_tokens=cfg.judge_max_tokens,
    )

    logger.info("Templating %s judge prompts (offline)...", f"{len(candidates):,}")
    templated = [
        tokenizer.apply_chat_template(
            _build_judge_messages(r),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for r in candidates
    ]

    all_outputs = []
    total_chunks = math.ceil(len(templated) / cfg.judge_chunk_size)
    for chunk_idx in range(total_chunks):
        start = chunk_idx * cfg.judge_chunk_size
        end = min(start + cfg.judge_chunk_size, len(templated))
        logger.info(
            "Judge chunk %d/%d (%s prompts)...",
            chunk_idx + 1, total_chunks, f"{end - start:,}",
        )
        outputs = llm.generate(templated[start:end], params)
        all_outputs.extend(outputs)

    for result, out in zip(candidates, all_outputs):
        raw = out.outputs[0].text.strip() if out.outputs else ""
        _apply_verdict(result, raw)


def _run_judge_http(
    candidates: list[FilterResult], cfg: FilterConfig
) -> None:
    """Judge candidates by hitting an OpenAI-compatible HTTP server.

    Reuses a running `vllm serve` instance, so we don't need to load the model
    in-process. Concurrent requests are bounded by cfg.judge_concurrency."""
    import asyncio
    import aiohttp

    api_url = f"{cfg.judge_base_url.rstrip('/')}/chat/completions"
    logger.info(
        "HTTP judge: %s candidates → %s (concurrency=%d)",
        f"{len(candidates):,}", api_url, cfg.judge_concurrency,
    )

    async def _one(session, sem, r: FilterResult) -> tuple[str | None, str]:
        """Returns (verdict, reason); verdict None means judge_error.

        Empty or unparseable judge output consumes a retry attempt just like
        an HTTP failure, so transient truncation/overload doesn't silently
        misclassify a record."""
        payload = {
            "model": cfg.model,
            "messages": _build_judge_messages(r),
            "temperature": cfg.judge_temperature,
            "max_tokens": cfg.judge_max_tokens,
        }
        last_reason = "no judge response"
        for attempt in range(3):
            try:
                async with sem:
                    async with session.post(
                        api_url, json=payload,
                        timeout=aiohttp.ClientTimeout(total=cfg.judge_request_timeout),
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        raw = (data["choices"][0]["message"]["content"] or "").strip()
            except Exception as e:
                logger.warning("Judge HTTP attempt %d failed: %s", attempt + 1, e)
                last_reason = f"judge request failed: {e}"
                await asyncio.sleep(2 ** attempt)
                continue
            verdict, reason = _parse_judge_verdict(raw)
            if verdict is not None:
                return verdict, reason
            last_reason = reason
            logger.warning("Judge attempt %d unparseable: %s", attempt + 1, reason)
        return None, last_reason

    async def _all():
        sem = asyncio.Semaphore(cfg.judge_concurrency)
        async with aiohttp.ClientSession() as session:
            tasks = [_one(session, sem, r) for r in candidates]
            # Progress in chunks so we get periodic log lines
            outs: list[tuple[str | None, str]] = []
            chunk = max(cfg.judge_chunk_size, cfg.judge_concurrency * 4)
            for i in range(0, len(tasks), chunk):
                batch = tasks[i:i + chunk]
                outs.extend(await asyncio.gather(*batch))
                logger.info(
                    "Judge progress: %s/%s",
                    f"{min(i + chunk, len(tasks)):,}", f"{len(tasks):,}",
                )
            return outs

    verdicts = asyncio.run(_all())
    for result, (verdict, reason) in zip(candidates, verdicts):
        _apply_parsed_verdict(result, verdict, reason)


def run_judge(results: list[FilterResult], cfg: FilterConfig) -> None:
    """Run the LLM judge on tier-1-passing records. Mutates results in place."""
    candidates = _select_judge_candidates(results, cfg)
    if not candidates:
        logger.info("No candidates for LLM judge.")
        return

    if cfg.judge_backend == "http_async":
        _run_judge_http(candidates, cfg)
    elif cfg.judge_backend == "offline":
        _run_judge_offline(candidates, cfg)
    else:
        raise ValueError(
            f"Unknown judge_backend: {cfg.judge_backend!r}. "
            "Expected 'offline' or 'http_async'."
        )

    n_err = sum(1 for r in candidates if r.judge_error)
    if n_err:
        logger.warning(
            "Judge: %s record(s) had empty/unparseable output after retries — "
            "held out as judge_error (written to *.errors.jsonl; excluded from "
            "both kept and removed sets).",
            f"{n_err:,}",
        )


_VERDICT_TOKEN_RE = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _parse_judge_verdict(text: str) -> tuple[str | None, str]:
    """Parse 'YES — reason' or 'NO — reason' from judge output.

    The judge runs with thinking enabled, so chain-of-thought is stripped
    first: only content after the FINAL </think> tag is considered. (When the
    server returns reasoning separately, the message content we receive here
    never contains it in the first place.) Returns (None, reason) when no
    verdict can be extracted — callers retry and, failing that, record an
    explicit judge_error; an unparseable output is never treated as "NO".
    """
    text = (text or "").strip()
    if not text:
        return None, "empty judge response"

    if _THINK_CLOSE in text:
        text = text.rsplit(_THINK_CLOSE, 1)[1].strip()
        if not text:
            return None, "no content after thinking block"
    elif _THINK_OPEN in text:
        # Opening tag without a closing tag: output truncated mid-thought,
        # everything present is chain-of-thought — there is no verdict.
        return None, "judge output truncated inside thinking block"

    # Fast path: verdict word is the very first token (what the prompt asks for).
    upper = text.upper()
    if upper.startswith("YES"):
        return "YES", text[3:].lstrip(" —-–:").strip()
    if upper.startswith("NO") and (len(text) < 3 or not text[2].isalpha()):
        return "NO", text[2:].lstrip(" —-–:").strip()

    # Slow path: scan for the first standalone YES/NO token in the
    # (post-thinking) answer. This catches judges that wrote their brief
    # explanation before the verdict word.
    m = _VERDICT_TOKEN_RE.search(text)
    if m is not None:
        verdict = m.group(1).upper()
        # Reason = whatever follows the verdict token, stripped of separators.
        tail = text[m.end():].lstrip(" —-–:").strip()
        if not tail:
            # Fall back to the preceding text as the reason — it's the
            # judge's explanation.
            tail = text[: m.start()].strip()
        return verdict, tail or text[:200]

    return None, f"ambiguous judge response: {text[:200]}"


# ---------------------------------------------------------------------------
# Output & stats
# ---------------------------------------------------------------------------

def compute_stats(results: list[FilterResult]) -> dict:
    """Per-file and overall stats."""
    by_file: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "kept": 0,
        "rule_removed": defaultdict(int),
        "judge_removed": 0,
        "judge_passed": 0,
        "judge_skipped": 0,
        "judge_error": 0,
    })
    for r in results:
        s = by_file[r.source_path]
        s["total"] += 1
        if r.judge_error:
            s["judge_error"] += 1
        elif r.kept:
            s["kept"] += 1
        if r.tier == "rule":
            s["rule_removed"][r.rule_name] += 1
        elif r.tier == "judge":
            s["judge_removed"] += 1
        # Track judge coverage
        if r.judge_verdict:
            if r.judge_verdict == "YES":
                s["judge_passed"] += 1
        elif r.tier != "rule" and not r.judge_error:
            s["judge_skipped"] += 1

    return dict(by_file)


def print_stats(stats: dict) -> None:
    overall_total = 0
    overall_kept = 0
    overall_rule = 0
    overall_judge = 0
    overall_error = 0

    for src_path, s in stats.items():
        fname = Path(src_path).name
        rule_total = sum(s["rule_removed"].values())
        logger.info("=" * 60)
        logger.info("File: %s", fname)
        logger.info("  Total:          %s", f"{s['total']:,}")
        logger.info("  Kept:           %s (%.1f%%)", f"{s['kept']:,}",
                     100 * s['kept'] / max(s['total'], 1))
        logger.info("  Rule-removed:   %s", f"{rule_total:,}")
        for rule_name, count in sorted(s["rule_removed"].items()):
            logger.info("    %-25s %s", rule_name, f"{count:,}")
        logger.info("  Judge-removed:  %s", f"{s['judge_removed']:,}")
        logger.info("  Judge-passed:   %s", f"{s['judge_passed']:,}")
        logger.info("  Judge-skipped:  %s", f"{s['judge_skipped']:,}")
        logger.info("  Judge-error:    %s", f"{s['judge_error']:,}")

        overall_total += s["total"]
        overall_kept += s["kept"]
        overall_rule += rule_total
        overall_judge += s["judge_removed"]
        overall_error += s["judge_error"]

    logger.info("=" * 60)
    logger.info("OVERALL")
    logger.info("  Total:          %s", f"{overall_total:,}")
    logger.info("  Kept:           %s (%.1f%%)", f"{overall_kept:,}",
                 100 * overall_kept / max(overall_total, 1))
    logger.info("  Rule-removed:   %s", f"{overall_rule:,}")
    logger.info("  Judge-removed:  %s", f"{overall_judge:,}")
    logger.info("  Judge-error:    %s", f"{overall_error:,}")


def write_outputs(results: list[FilterResult], cfg: FilterConfig) -> None:
    """Write filtered JSONL files and removal log."""
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by source file
    grouped: OrderedDict[str, list[FilterResult]] = OrderedDict()
    for r in results:
        grouped.setdefault(r.source_path, []).append(r)

    for src_path, file_results in grouped.items():
        fname = Path(src_path).name

        # Write filtered JSONL (kept records only; judge_error records are
        # in neither the kept nor the removed set — they go to .errors.jsonl)
        filtered_path = out_dir / fname
        with open(filtered_path, "w", encoding="utf-8") as f:
            for r in file_results:
                if r.kept and not r.judge_error:
                    f.write(json.dumps(r.record, ensure_ascii=False) + "\n")
        kept_count = sum(1 for r in file_results if r.kept and not r.judge_error)
        logger.info("Wrote %s kept records to %s", f"{kept_count:,}", filtered_path)

        # Write removal log
        log_path = out_dir / fname.replace(".jsonl", ".removed.jsonl")
        with open(log_path, "w", encoding="utf-8") as f:
            for r in file_results:
                if not r.kept:
                    entry = {
                        "id": r.record.get("id", ""),
                        "tier": r.tier,
                        "rule_name": r.rule_name,
                        "judge_verdict": r.judge_verdict,
                        "judge_reason": r.judge_reason,
                        "prompt": r.record.get("prompt", "")[:500],
                        "instruction_set": r.record.get("instruction_set", "")[:500],
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        removed_count = sum(1 for r in file_results if not r.kept)
        logger.info("Wrote %s removal log entries to %s", f"{removed_count:,}", log_path)

        # Judge-error sidecar: judge output stayed empty/unparseable after all
        # retries. Held out of both kept and removed so a rerun can re-judge.
        error_results = [r for r in file_results if r.judge_error]
        if error_results:
            err_path = out_dir / fname.replace(".jsonl", ".errors.jsonl")
            with open(err_path, "w", encoding="utf-8") as f:
                for r in error_results:
                    entry = {
                        "id": r.record.get("id", ""),
                        "judge_error": r.judge_error,
                        "prompt": r.record.get("prompt", "")[:500],
                        "instruction_set": r.record.get("instruction_set", "")[:500],
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.info("Wrote %s judge-error entries to %s",
                        f"{len(error_results):,}", err_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_records(paths: list[str]) -> list[tuple[str, dict]]:
    records = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append((path, json.loads(line)))
    logger.info("Loaded %s records from %d files", f"{len(records):,}", len(paths))
    return records


def build_tokenizer(model: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def main():
    parser = argparse.ArgumentParser(
        description="Filter the instruction-set dataset: rule-based + LLM judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", nargs="+", required=True,
                        help="Input JSONL file globs")
    parser.add_argument("--output-dir", default="filtered",
                        help="Directory for filtered files and logs (default: filtered/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report stats only, do not write files")
    parser.add_argument("--rules-only", action="store_true",
                        help="Skip LLM judge, run only rule-based filtering")
    parser.add_argument("--judge-sample", type=float, default=1.0,
                        help="Fraction of tier-1-passing records to judge (0.0-1.0, default: 1.0)")
    parser.add_argument("--seed", type=int, default=42)

    # LLM settings
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--judge-max-tokens", type=int, default=None,
                        help="Max new tokens for the judge (default: use "
                             "FilterConfig.judge_max_tokens = 8192, which leaves "
                             "room for the thinking block + verdict)")
    parser.add_argument("--judge-chunk-size", type=int, default=2000)
    parser.add_argument("--judge-backend", choices=["offline", "http_async"],
                        default="offline",
                        help="'offline' loads vLLM in-process; 'http_async' hits a running server.")
    parser.add_argument("--judge-base-url", default="http://localhost:8089/v1",
                        help="OpenAI-compatible base URL (only for http_async)")
    parser.add_argument("--judge-concurrency", type=int, default=32,
                        help="Concurrent judge requests (http_async only)")
    parser.add_argument("--judge-temperature", type=float, default=0.0,
                        help="Judge sampling temperature (http_async only; offline uses fixed params)")

    # Rule thresholds
    parser.add_argument("--min-response-len", type=int, default=10,
                        help="Min chars for instruction_set (default: 10)")
    parser.add_argument("--prompt-b-overlap-threshold", type=float, default=0.75,
                        help="Max word overlap with retrieval_prompt before flagging as echo (default: 0.75)")

    args = parser.parse_args()

    cfg = FilterConfig(
        input_globs=args.input or [
            "prompt_only_instruction_set_dataset*.jsonl",
            "prompt_only_oracle_dataset*.jsonl",  # legacy
        ],
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        rules_only=args.rules_only,
        judge_sample=args.judge_sample,
        seed=args.seed,
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        judge_max_tokens=(args.judge_max_tokens if args.judge_max_tokens is not None
                          else FilterConfig.judge_max_tokens),
        judge_chunk_size=args.judge_chunk_size,
        judge_backend=args.judge_backend,
        judge_base_url=args.judge_base_url,
        judge_concurrency=args.judge_concurrency,
        judge_temperature=args.judge_temperature,
        min_response_len=args.min_response_len,
        prompt_b_overlap_threshold=args.prompt_b_overlap_threshold,
    )

    # Resolve input files. Globs are expanded; explicit paths pass through.
    # We exclude files already inside the output dir (avoids self-feeding when
    # the user runs a bare glob like "*.jsonl" from a project root that also
    # contains `filtered/`).
    out_abs = os.path.abspath(cfg.output_dir)
    paths_set: set[str] = set()
    for pattern in cfg.input_globs:
        matched = glob.glob(pattern) or [pattern]
        for p in matched:
            if not os.path.exists(p):
                continue
            if os.path.abspath(p).startswith(out_abs + os.sep):
                continue
            paths_set.add(p)
    paths = sorted(paths_set)

    if not paths:
        logger.error("No input files found.")
        return
    logger.info("Input files: %s", paths)
    logger.info("Config: dry_run=%s, rules_only=%s, judge_sample=%.0f%%",
                 cfg.dry_run, cfg.rules_only, cfg.judge_sample * 100)

    records = load_records(paths)

    # -- Tier 1: rule-based --
    results: list[FilterResult] = []
    for src_path, rec in records:
        passed, rule_name = apply_rules(rec, cfg)
        fr = FilterResult(record=rec, source_path=src_path)
        if not passed:
            fr.kept = False
            fr.tier = "rule"
            fr.rule_name = rule_name
        results.append(fr)

    rule_removed = sum(1 for r in results if not r.kept)
    logger.info("Tier 1 (rules): %s removed, %s remaining",
                 f"{rule_removed:,}", f"{len(results) - rule_removed:,}")

    # -- Tier 2: LLM judge --
    if not cfg.rules_only:
        run_judge(results, cfg)
        judge_removed = sum(1 for r in results if r.tier == "judge")
        logger.info("Tier 2 (judge): %s removed", f"{judge_removed:,}")

    # -- Stats --
    stats = compute_stats(results)
    print_stats(stats)

    # -- Write output --
    if not cfg.dry_run:
        write_outputs(results, cfg)
    else:
        logger.info("Dry run — no files written.")


if __name__ == "__main__":
    main()

