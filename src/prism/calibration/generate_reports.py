"""generate_reports.py — greedy SFT-decoder reports for the calibration set.

Reads `pilot_ids.json` and `main_ids.json` (from sample.py), loads the SFT
LoRA + projection, generates ONE greedy report per record via the
monitor pipeline (raw activations → projection →
norm_match → inputs_embeds → target-model generate).

Outputs:
  sft_reports.jsonl — one JSON line per record:
    {"record_id", "global_idx", "source_dataset", "phase" ("pilot" | "main"),
     "prompt_a", "response_a", "response_b", "sft_report"}

Run:
    uv run python -m prism.calibration.generate_reports --in-dir calibration_out
    uv run python -m prism.calibration.generate_reports --in-dir calibration_out --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType
from torch.utils.data import DataLoader, Subset

from prism.activations import PrecomputedActivationDataset, build_precomputed_collate_fn
from prism.common.env import load_env
from prism.rl import rollouts
from prism.rl.adapters import POLICY_ADAPTER, ProjectionPair, attach_policy_and_ref
from prism.rl.config import RL_CONFIG
from prism.rl.data import build_prefix_embeddings
from prism.sft.model import compute_target_norm, norm_match

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _load_ids(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def main():
    from prism import target_models
    if target_models.active_name() != "qwen3.5-9b":
        raise SystemExit(
            "This tool loads the target model with Qwen-specific classes; it "
            "currently supports only PRISM_TARGET_MODEL=qwen3.5-9b."
        )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=str, required=True,
                   help="Directory containing pilot_ids.json + main_ids.json (from sample.py)")
    p.add_argument("--out", type=str, default=None,
                   help="Output JSONL path (default: <in-dir>/sft_reports.jsonl)")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Generation batch size (greedy, no sampling — bigger is fine)")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Override cfg gen_max_new_tokens (default uses cfg)")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after this many records (smoke testing)")
    args = p.parse_args()

    load_env()
    cfg = dict(RL_CONFIG)
    if cfg.get("judge_model", "MODEL_NOT_SET") == "MODEL_NOT_SET":
        cfg["judge_model"] = os.environ.get("PRISM_JUDGE_MODEL") or ""
    cfg["batch_size"] = args.batch_size
    if args.max_new_tokens is not None:
        cfg["gen_max_new_tokens"] = args.max_new_tokens
    device = "cuda" if torch.cuda.is_available() else "cpu"

    in_dir = Path(args.in_dir)
    pilot = _load_ids(in_dir / "pilot_ids.json")
    main_ids = _load_ids(in_dir / "main_ids.json")
    logger.info("Loaded %d pilot + %d main records", len(pilot), len(main_ids))

    all_records = [(rec, "pilot") for rec in pilot] + [(rec, "main") for rec in main_ids]
    if args.limit is not None:
        all_records = all_records[: args.limit]
    logger.info("Will generate %d reports total", len(all_records))

    # ── Load Qwen + dual LoRA + ProjectionPair ───────────────────────────────
    logger.info("Loading %s …", cfg["model_id"])
    t0 = time.time()
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(cfg["model_id"])
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModelForImageTextToText.from_pretrained(
        cfg["model_id"], dtype=torch.bfloat16, device_map=device,
    )
    qwen.config.use_cache = False
    qwen.eval()
    for p_ in qwen.parameters():
        p_.requires_grad_(False)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
    )
    qwen = attach_policy_and_ref(qwen, lora_cfg, cfg["sft_init_from"], device)
    qwen.set_adapter(POLICY_ADAPTER)
    pair = ProjectionPair(
        dim=cfg["projection_dim"], sft_init_from=cfg["sft_init_from"],
        device=device, dtype=torch.bfloat16,
    )
    pair.policy.eval()
    target_norm = compute_target_norm(qwen).to(device)
    logger.info("Setup done in %.1fs", time.time() - t0)

    # ── Build a Subset of the test split over chosen global_idxs ─────────────
    hook_layer = cfg["hook_layer"]
    ds = PrecomputedActivationDataset(cfg["precomputed_dir"], split="test", layers=[hook_layer])
    indices = [rec["global_idx"] for rec, _ in all_records]
    subset = Subset(ds, indices)
    # We carry "phase" alongside, mapped by position in `all_records`.
    phase_by_pos = [phase for _, phase in all_records]
    rec_by_pos = [rec for rec, _ in all_records]

    collate = build_precomputed_collate_fn(tokenizer, cfg, ds.source_map, layers=[hook_layer], training=False)
    loader = DataLoader(
        subset, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=collate, num_workers=2,
    )

    # ── Generate ─────────────────────────────────────────────────────────────
    out_path = Path(args.out) if args.out else in_dir / "sft_reports.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    pos = 0  # tracks position in `all_records` to attach phase / record_id

    t_gen0 = time.time()
    with out_path.open("w") as fh, torch.no_grad():
        for batch in loader:
            act = batch["precomputed_acts"][hook_layer].to(device)
            act_mask = batch["act_masks"].to(device)
            chat_pref_ids = batch["chat_prefix_input_ids"].to(device)
            chat_pref_mask = batch["chat_prefix_attention_mask"].to(device)

            pol = pair.policy(act)
            pol_scaled = norm_match(pol, target_norm) * act_mask.unsqueeze(-1).to(pol.dtype)

            prefix_embeds, prefix_mask, _ = build_prefix_embeddings(
                pol_scaled, act_mask, chat_pref_ids, chat_pref_mask, qwen,
            )
            texts_by_prompt, _ = rollouts.generate_candidates(
                qwen, tokenizer,
                prefix_embeds=prefix_embeds, prefix_attention_mask=prefix_mask,
                n_candidates=1, max_new_tokens=cfg["gen_max_new_tokens"],
                temperature=0.0, top_p=1.0, do_sample=False,
            )

            B = act.shape[0]
            n_written_batch = 0
            for b in range(B):
                rec = rec_by_pos[pos + b]
                phase = phase_by_pos[pos + b]
                if b >= len(texts_by_prompt) or not texts_by_prompt[b]:
                    logger.warning("Empty candidates for record %s (phase=%s, src=%s) — skipping",
                                   rec["record_id"], phase, rec["source_dataset"])
                    continue
                row = {
                    "record_id": rec["record_id"],
                    "global_idx": rec["global_idx"],
                    "source_dataset": rec["source_dataset"],
                    "phase": phase,
                    "prompt_a": batch["prompts_a"][b],
                    "response_a": batch["responses_a"][b],
                    "response_b": batch["ground_truth_b"][b],
                    "sft_report": texts_by_prompt[b][0],
                }
                fh.write(json.dumps(row) + "\n")
                n_written_batch += 1
            pos += B
            n_written += n_written_batch
            if n_written % max(1, args.batch_size * 4) == 0:
                logger.info("  generated %d / %d (%.1fs/record)",
                            n_written, len(all_records),
                            (time.time() - t_gen0) / max(1, n_written))

    logger.info("Wrote %d reports → %s in %.1fs", n_written, out_path, time.time() - t_gen0)


if __name__ == "__main__":
    main()
