#!/usr/bin/env python
"""Verify ON-THE-FLY activation extraction against an activation cache.

Loads the target model exactly as prism.rl.train does (profile loader, PEFT
wrapper with adapters disabled, early-exit hook at the hook layer), extracts
activations in-loop for N records of a cache split, and compares them with
the cached tensors. Also asserts that the on-the-fly collate produces the same
decoder prefix as the precomputed collate, and that the JSONL loader finds
the same records (id + text) as the cache metadata.

    uv run python scripts/check_onthefly_parity.py \
        --precomputed-dir $PRISM_DATA_DIR/precomputed/qwen3.5-9b-L16-prompt-only \
        --dataset-paths $PRISM_DATA_DIR/prompt-only/jsonl/*.jsonl --n 64

Exit code 1 on any structural mismatch (split membership, prefix ids, record
lookup; response token-count mismatches — tokenizer drift between the
transformers version that built the cache and the current one — are counted
and only fail with --strict-tokens). Numerical differences are reported (max |Δ|, mean |Δ|, min cosine),
and fail only if --max-abs-tol / --min-cos are given and violated. Expect
kernel-order noise (~1e-2 in bf16) for models whose extraction class equals
the training class (gemma-2, Ministral); for Qwen3.5 the released cache was
extracted with AutoModelForCausalLM while training uses
AutoModelForImageTextToText (docs/KNOWN_ISSUES.md) — this script quantifies
that gap.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import torch
from peft import LoraConfig, TaskType, get_peft_model

from prism.common.env import load_env
from prism.activations import PrecomputedActivationDataset, build_onthefly_collate_fn, build_precomputed_collate_fn
from prism.activations.records import Record, load_records
from prism.rl.config import RL_CONFIG
from prism.rl.train import _extract_batch_activations
from prism.sft.model import register_hook

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("parity")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--precomputed-dir", required=True)
    ap.add_argument("--dataset-paths", nargs="*", default=[], help="JSONL files (optional loader cross-check)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-abs-tol", type=float, default=None)
    ap.add_argument("--min-cos", type=float, default=None)
    ap.add_argument("--strict-tokens", action="store_true", help="fail on any response token-count mismatch")
    args = ap.parse_args()
    load_env()

    cfg = dict(RL_CONFIG)
    cfg["precomputed_dir"] = args.precomputed_dir
    hook_layer = cfg["hook_layer"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from prism.target_models import load_target_model
    t0 = time.time()
    model, tokenizer, _profile = load_target_model(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    act_store, _hook = register_hook(model, hook_layer)          # same order as rl.train
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=cfg["lora_r"],
                                             lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
                                             target_modules=cfg["lora_target_modules"]))
    log.info("model ready in %.1fs (hook layer %d, class %s)", time.time() - t0, hook_layer, type(model.base_model.model).__name__)

    ds = PrecomputedActivationDataset(args.precomputed_dir, split=args.split, layers=[hook_layer])
    n = min(args.n, len(ds))
    samples = [ds[i] for i in range(n)]
    cfg["_target_profile"] = os.environ.get("PRISM_TARGET_MODEL")
    otf_collate = build_onthefly_collate_fn(tokenizer, cfg, ds.source_map)
    pre_collate = build_precomputed_collate_fn(tokenizer, cfg, ds.source_map, layers=[hook_layer], training=False)

    # optional: JSONL loader must find the same records with the same text
    by_id = {}
    if args.dataset_paths:
        recs, _ = load_records(args.dataset_paths)
        by_id = {r.id: r for r in recs}
        miss = [s["record_id"] for s in samples if s["record_id"] not in by_id]
        bad = [s["record_id"] for s in samples if s["record_id"] in by_id and
               (by_id[s["record_id"]].prompt_a != s["prompt_a"] or by_id[s["record_id"]].response_a != s["response_a"]
                or by_id[s["record_id"]].response_b != s["response_b"])]
        log.info("loader cross-check: %d/%d ids found, %d text mismatches", n - len(miss), n, len(bad))
        if miss or bad:
            log.error("loader mismatch: missing=%s bad=%s", miss[:5], bad[:5])
            return 1

    fails = 0
    # split-membership check: the on-the-fly split of the same files must equal the cache split
    if args.dataset_paths:
        from prism.activations.records import load_split_records
        from prism.rl.sampler import enumerate_record_ids
        scfg = dict(cfg); scfg["dataset_paths"] = list(args.dataset_paths)
        otf_tr, otf_va, otf_te, _ = load_split_records(scfg, None)
        for name, part in (("train", otf_tr), ("val", otf_va), ("test", otf_te)):
            try:
                cds = PrecomputedActivationDataset(args.precomputed_dir, split=name, layers=[hook_layer])
            except Exception as e:  # split absent in cache
                log.info("split %s: cache has no such split (%s)", name, e); continue
            cache_ids = set(enumerate_record_ids(cds))
            otf_ids = set(r.id for r in part)
            log.info("split %s: cache=%d on-the-fly=%d common=%d", name, len(cache_ids), len(otf_ids), len(cache_ids & otf_ids))
            if cache_ids != otf_ids:
                log.error("split %s membership differs (only-cache=%d only-otf=%d)", name, len(cache_ids - otf_ids), len(otf_ids - cache_ids)); fails += 1
    max_abs = 0.0; sum_abs = 0.0; n_elem = 0; min_cos = 1.0; extract_s = 0.0; n_tok_mismatch = 0
    for start in range(0, n, args.batch_size):
        chunk = samples[start:start + args.batch_size]
        recs = [Record(s["record_id"], s["source_dataset"], s["prompt_a"], s["response_a"], "", s["response_b"]) for s in chunk]
        otf = otf_collate(recs)
        pre = pre_collate(chunk)
        if not torch.equal(otf["chat_prefix_input_ids"], pre["chat_prefix_input_ids"]):
            log.error("decoder prefix ids differ at batch %d", start); fails += 1
        t = time.time()
        act, mask = _extract_batch_activations(model, act_store, otf, cfg, device)
        torch.cuda.synchronize() if device == "cuda" else None
        extract_s += time.time() - t
        cached = pre["precomputed_acts"][hook_layer].to(device)
        cmask = pre["act_masks"].to(device)
        for b in range(len(chunk)):
            n_otf = int(mask[b].sum()); n_pre = int(cmask[b].sum())
            if n_otf != n_pre:
                # Usually tokenizer drift between the transformers version that
                # built the cache and the current one (content-specific, rare).
                n_tok_mismatch += 1
                log.warning("%s: response token count on-the-fly=%d cache=%d (skipped)", chunk[b]["record_id"], n_otf, n_pre)
                continue
            a = act[b, :n_otf].float(); c = cached[b, :n_pre].float()
            d = (a - c).abs()
            max_abs = max(max_abs, float(d.max())); sum_abs += float(d.sum()); n_elem += d.numel()
            cos = torch.nn.functional.cosine_similarity(a, c, dim=-1)
            min_cos = min(min_cos, float(cos.min()))
    log.info("records=%d (token-count mismatches: %d) | max|Δ|=%.4g mean|Δ|=%.4g min cos=%.6f | extraction %.3fs/batch of %d",
             n, n_tok_mismatch, max_abs, sum_abs / max(n_elem, 1), min_cos, extract_s / max(1, (n + args.batch_size - 1) // args.batch_size), args.batch_size)
    if args.max_abs_tol is not None and max_abs > args.max_abs_tol:
        log.error("max|Δ| %.4g > tol %.4g", max_abs, args.max_abs_tol); fails += 1
    if args.strict_tokens and n_tok_mismatch:
        log.error("%d token-count mismatches", n_tok_mismatch); fails += 1
    if args.min_cos is not None and min_cos < args.min_cos:
        log.error("min cos %.6f < %.6f", min_cos, args.min_cos); fails += 1
    print("PARITY", "FAIL" if fails else "OK", f"max_abs={max_abs:.4g} mean_abs={sum_abs / max(n_elem, 1):.4g} min_cos={min_cos:.6f}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
