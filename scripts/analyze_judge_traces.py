"""Bin judge_traces.jsonl by training step and decompose reward.

Usage: python3 analyze_judge_traces.py <trace.jsonl> [bin_size]
Streams the file; prints per-bin means of reward components plus
bullet-count stats and judge-error rate.
"""
import argparse
import json
from collections import defaultdict

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("trace", help="<checkpoint_dir>/judge_traces.jsonl written by prism.rl.train")
_ap.add_argument("bin_size", nargs="?", type=int, default=500, help="steps per bin (default 500)")
_args = _ap.parse_args()
path, bin_size = _args.trace, _args.bin_size

bins = defaultdict(lambda: {
    "n": 0, "reward": 0.0, "inst": 0.0, "halluc": 0.0, "lp": 0.0,
    "n_bullets": 0, "n_gt": 0, "err": 0, "neg": 0, "halluc_nonzero": 0,
})

with open(path) as f:
    for line in f:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        b = bins[r["step"] // bin_size]
        b["n"] += 1
        b["reward"] += r["reward"]
        b["inst"] += r["mean_instruction_score"]
        b["halluc"] += r["mean_hallucination_score"]
        b["lp"] += r["length_penalty"]
        b["n_bullets"] += len(r.get("hallucination_scores") or [])
        b["n_gt"] += len(r.get("instruction_scores") or [])
        b["err"] += 1 if r.get("judge_error") else 0
        b["neg"] += 1 if r["reward"] < 0 else 0
        b["halluc_nonzero"] += 1 if r["mean_hallucination_score"] > 0 else 0

print(f"{'steps':>12} {'n':>6} {'reward':>7} {'inst':>6} {'halluc':>7} "
      f"{'lenpen':>7} {'bullets':>7} {'gt':>5} {'neg%':>5} {'hal>0%':>6} {'err%':>5}")
for k in sorted(bins):
    b = bins[k]
    n = b["n"]
    print(f"{k*bin_size:>5}-{(k+1)*bin_size:<6} {n:>6} "
          f"{b['reward']/n:>7.3f} {b['inst']/n:>6.3f} {b['halluc']/n:>7.3f} "
          f"{b['lp']/n:>7.3f} {b['n_bullets']/n:>7.2f} {b['n_gt']/n:>5.2f} "
          f"{100*b['neg']/n:>4.1f} {100*b['halluc_nonzero']/n:>6.1f} "
          f"{100*b['err']/n:>4.1f}")
