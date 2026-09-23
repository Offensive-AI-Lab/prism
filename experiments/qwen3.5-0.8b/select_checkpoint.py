#!/usr/bin/env python3
"""Score a prism-eval probe output for BOTH coverage and report FAITHFULNESS.

Coverage alone is a lenient proxy — it credits a generic paraphrase as much as a
crisp, constraint-retaining report (that is exactly how the step-14600 "collapse"
kept ~0.55 coverage while dropping specifics). So for checkpoint selection we also
measure the collapse signature directly: the fraction of reports that open with the
templated generic opener, plus mean report length.

Pick the checkpoint that is FAITHFUL (low generic-opener fraction) AND ~target coverage.

Usage:
  select_checkpoint.py <probe_results_dir> [<probe_results_dir> ...]
  where each dir contains prism_08b_ckpt_probe/{rows.jsonl,summary.json}
  (or pass a rows.jsonl directly).
"""
from __future__ import annotations
import json, sys, re
from pathlib import Path

GENERIC = re.compile(r"^[\-\*\s]*answer the provided question or instruction", re.I)


def report_of(row: dict) -> str:
    o = row.get("output", row)
    if isinstance(o, dict):
        return o.get("itm_report") or o.get("report") or ""
    return str(o)


def cov_of(row: dict):
    s = row.get("scores", {})
    j = s.get("JudgeLLMScorer", {}) if isinstance(s, dict) else {}
    return j.get("coverage", j.get("instruction_score"))


def find_rows(arg: str) -> Path:
    p = Path(arg)
    if p.is_file():
        return p
    for c in (p / "prism_08b_ckpt_probe" / "rows.jsonl", p / "rows.jsonl"):
        if c.exists():
            return c
    # any rows.jsonl under the dir
    hits = list(p.rglob("rows.jsonl"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"no rows.jsonl under {arg}")


def score(rows_path: Path) -> dict:
    rows = [json.loads(l) for l in open(rows_path) if l.strip()]
    n = len(rows)
    generic = sum(1 for r in rows if GENERIC.match(report_of(r).lstrip()))
    lens = [len(report_of(r)) for r in rows]
    covs = [c for c in (cov_of(r) for r in rows) if c is not None]
    # per-setting generic fraction
    by = {}
    for r in rows:
        st = r.get("setting", "?")
        b = by.setdefault(st, {"n": 0, "gen": 0})
        b["n"] += 1
        b["gen"] += int(bool(GENERIC.match(report_of(r).lstrip())))
    return {
        "path": str(rows_path),
        "n": n,
        "coverage": round(sum(covs) / len(covs), 4) if covs else None,
        "generic_opener_frac": round(generic / n, 4) if n else None,
        "mean_report_len": round(sum(lens) / n, 1) if n else None,
        "per_setting_generic": {k: round(v["gen"] / v["n"], 3) for k, v in sorted(by.items())},
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    results = [score(find_rows(a)) for a in sys.argv[1:]]
    print(f"{'label':<28} {'cov':>6} {'generic%':>9} {'len':>7}   per-setting-generic")
    for r in results:
        label = Path(r["path"]).parts[-3] if "prism_08b" in r["path"] else Path(r["path"]).parent.name
        print(f"{label:<28} {r['coverage']!s:>6} {r['generic_opener_frac']!s:>9} "
              f"{r['mean_report_len']!s:>7}   {r['per_setting_generic']}")
    print("\nPick: faithful = LOW generic% (SFT/pre-collapse ~0.0), coverage ~target.")


if __name__ == "__main__":
    main()
