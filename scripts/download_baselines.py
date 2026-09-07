#!/usr/bin/env python3
"""Download the demo's baseline LoRA adapters (LatentQA, Activation Oracles).

The interactive demo's compare mode auto-downloads and SHA-256-verifies these
on first use, but that path needs a GPU (it loads the target model first).
This standalone tool fetches and verifies them without a GPU — useful to
pre-stage or to check the artifacts on their own.

    python scripts/download_baselines.py                 # dest = ./checkpoints/baselines
    python scripts/download_baselines.py --dest DIR --only latentqa
    python scripts/download_baselines.py --verify-only

Downloads are verified against the SHA256 manifest below; a mismatch is a hard
error. The repos and digests are the same ones the demo pins in
demo/baselines.py (keep the two in sync).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

ORG = os.environ.get("PRISM_DEMO_WEIGHTS_ORG", "Offensive-AI-Lab")

# name -> (repo, {filename: sha256}). Mirrors BASELINE_ADAPTERS in demo/baselines.py.
MANIFEST: dict[str, tuple[str, dict[str, str]]] = {
    "latentqa": ("prism-baseline-latentqa-qwen3.5-9b", {
        "adapter_config.json": "a952a74b6be55979834a491a7da2f1ddfff0f9ddc153116b692c6b6ab4a220a6",
        "adapter_model.safetensors": "8600f3ba51e60e53de72ff044adee962dca3c179a9b8a1212809c236a7852cd2",
    }),
    "ao": ("prism-baseline-activation-oracles-qwen3.5-9b", {
        "adapter_config.json": "b049c48d32dfbb25e605949b6859dbe2eb53bd680d36eb6171f82e97003306cc",
        "adapter_model.safetensors": "1830598a70e652d4bf5a39de4439d7683e8e5825cc5c053b66cdcbbe187e1612",
    }),
}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path,
                    default=Path(os.environ.get("PRISM_DEMO_BASELINES_DIR", "checkpoints/baselines")),
                    help="Directory for the adapters (default: ./checkpoints/baselines)")
    ap.add_argument("--only", action="append", metavar="NAME",
                    help=f"Fetch only these (repeatable). Choices: {', '.join(MANIFEST)}")
    ap.add_argument("--verify-only", action="store_true",
                    help="Don't download; just checksum what's already in --dest.")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF token, if a repo is gated (default: $HF_TOKEN)")
    args = ap.parse_args()

    wanted = dict(MANIFEST)
    if args.only:
        unknown = set(args.only) - set(MANIFEST)
        if unknown:
            print(f"unknown --only: {unknown}; choices {list(MANIFEST)}", file=sys.stderr)
            return 2
        wanted = {k: v for k, v in MANIFEST.items() if k in args.only}

    fails = 0
    for name, (repo, files) in wanted.items():
        target_dir = args.dest / name
        target_dir.mkdir(parents=True, exist_ok=True)
        for fname, expected in files.items():
            path = target_dir / fname
            if not args.verify_only and not path.exists():
                from huggingface_hub import hf_hub_download
                print(f"→ {ORG}/{repo}/{fname}")
                hf_hub_download(repo_id=f"{ORG}/{repo}", filename=fname,
                                local_dir=str(target_dir), token=args.token)
            if not path.exists():
                print(f"  ✗ {name}/{fname}: missing", file=sys.stderr); fails += 1; continue
            actual = sha256_of(path)
            if actual == expected:
                print(f"  ✓ {name}/{fname}: checksum OK")
            else:
                print(f"  ✗ {name}/{fname}: {actual} != {expected}", file=sys.stderr); fails += 1

    if fails:
        print(f"\n{fails} problem(s).", file=sys.stderr); return 1
    print(f"\nBaseline adapters ready under {args.dest}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
