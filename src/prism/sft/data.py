"""
data.py — Dataset loading from JSONL files for activation-conditioned finetuning.

Loads multiple JSONL files, tags each sample with its source_dataset,
provides train/val splitting and a collate_fn that tokenizes retrieval_prompt/instruction_set.
"""

import json
import random
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class FinetuneRecord:
    id: str
    source_dataset: str
    prompt: str
    response: str
    instruction_set: str


def build_collate_fn(tokenizer, cfg: dict, source_map: dict, training: bool = True):
    """
    Returns a collate function that prepares a batch for training or validation.

    Args:
        training: If True and use_retrieval_prompt_variations is enabled, randomly
                  replaces retrieval_prompt with a variation. If False, always uses
                  the original retrieval_prompt from the JSONL for consistent eval.

    Each batch item becomes:
      - prompt_a_ids:  tokenized [prompt + response] for activation extraction
      - prompt_a_only_len: token count of prompt alone (to find response boundary)
      - chat_ids:      tokenized [retrieval_prompt + instruction_set] for decoder
      - answer_start:  index where instruction_set begins in chat_ids
      - source_id:     int dataset label
      - instruction_sets: list of instruction_set strings (for sample logging)
      - retrieval_prompts_used: retrieval prompts actually used (for sample logging)

    The collate_fn handles padding to batch-max lengths.
    """
    max_act_tokens = cfg["max_act_tokens"]
    max_target_len = cfg["max_target_len"]
    skip_prompt_b = cfg.get("skip_prompt_b", False)
    use_variations = training and cfg.get("use_retrieval_prompt_variations", False) and not skip_prompt_b
    variations = cfg.get("retrieval_prompt_variations", [])
    # Apply the target-model profile's system-message override exactly like the
    # precompute path (extract.py) does, so on-the-fly activations match the
    # precomputed ones (no-op for qwen/gemma; ministral suppresses its
    # ~530-token default system prompt).
    from prism.target_models import get_profile, wrap_messages
    _profile = get_profile(cfg.get("_target_profile"))

    def _apply_template(messages, **kwargs):
        """apply_chat_template wrapper that always returns a list of ints."""
        messages = wrap_messages(messages, _profile)
        try:
            result = tokenizer.apply_chat_template(
                messages, tokenize=True, enable_thinking=False, **kwargs
            )
        except TypeError:  # tokenizer template without an enable_thinking kwarg
            result = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
        if hasattr(result, "input_ids"):
            result = result["input_ids"]
        return result

    def collate_fn(batch: List[FinetuneRecord]):
        # ── Tokenize prompt + response for activation extraction ──────
        prompt_a_only_ids_list = []
        combined_a_ids_list = []
        response_a_token_counts = []

        for record in batch:
            prompt_a_only = _apply_template(
                [{"role": "user", "content": record.prompt}],
                add_generation_prompt=True,
            )
            combined_a = _apply_template(
                [
                    {"role": "user", "content": record.prompt},
                    {"role": "assistant", "content": record.response},
                ],
                add_generation_prompt=False,
            )

            resp_a_len = len(combined_a) - len(prompt_a_only)
            actual_act_len = min(resp_a_len, max_act_tokens)
            response_a_token_counts.append(actual_act_len)

            prompt_a_only_ids_list.append(len(prompt_a_only))
            combined_a_ids_list.append(torch.tensor(combined_a, dtype=torch.long))

        # Pad combined_a sequences for batch Qwen forward
        max_a_len = max(t.size(0) for t in combined_a_ids_list)
        padded_a_ids = torch.full(
            (len(batch), max_a_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        a_attention_mask = torch.zeros(len(batch), max_a_len, dtype=torch.long)
        for i, ids in enumerate(combined_a_ids_list):
            padded_a_ids[i, :ids.size(0)] = ids
            a_attention_mask[i, :ids.size(0)] = 1

        # ── Tokenize decoder target ───────────────────────────────────────
        chat_ids_list = []
        answer_start_list = []
        ground_truth_texts = []
        retrieval_prompts_used = []

        for record in batch:
            ground_truth_texts.append(record.instruction_set)

            if skip_prompt_b:
                # Pass empty-string user message to satisfy the chat template,
                # then set answer_start past the prefix so only instruction_set gets loss.
                retrieval_prompts_used.append("")
                prefix_ids = _apply_template(
                    [{"role": "user", "content": ""}],
                    add_generation_prompt=True,
                )
                full_ids = _apply_template(
                    [
                        {"role": "user", "content": ""},
                        {"role": "assistant", "content": record.instruction_set},
                    ],
                    add_generation_prompt=False,
                )
                if len(full_ids) > max_target_len:
                    full_ids = full_ids[:max_target_len]
                chat_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
                answer_start_list.append(len(prefix_ids))
            else:
                # Choose retrieval_prompt: variation (training) or original (eval)
                if use_variations and variations:
                    retrieval_prompt = random.choice(variations)
                else:
                    retrieval_prompt = cfg.get("decoder_user_prompt", "")

                retrieval_prompts_used.append(retrieval_prompt)

                prefix_ids = _apply_template(
                    [{"role": "user", "content": retrieval_prompt}],
                    add_generation_prompt=True,
                )
                full_ids = _apply_template(
                    [
                        {"role": "user", "content": retrieval_prompt},
                        {"role": "assistant", "content": record.instruction_set},
                    ],
                    add_generation_prompt=False,
                )
                if len(full_ids) > max_target_len:
                    full_ids = full_ids[:max_target_len]
                chat_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
                answer_start_list.append(len(prefix_ids))

        # Pad chat sequences
        max_chat_len = max(t.size(0) for t in chat_ids_list)
        padded_chat_ids = torch.full(
            (len(batch), max_chat_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        chat_attention_mask = torch.zeros(len(batch), max_chat_len, dtype=torch.long)
        chat_lengths = []
        for i, ids in enumerate(chat_ids_list):
            padded_chat_ids[i, :ids.size(0)] = ids
            chat_attention_mask[i, :ids.size(0)] = 1
            chat_lengths.append(ids.size(0))

        # ── Source dataset IDs ────────────────────────────────────────────
        source_ids = torch.tensor(
            [source_map.get(r.source_dataset, -1) for r in batch],
            dtype=torch.long,
        )

        return {
            # For Qwen activation extraction
            "a_input_ids": padded_a_ids,
            "a_attention_mask": a_attention_mask,
            "a_prompt_only_lens": torch.tensor(prompt_a_only_ids_list, dtype=torch.long),
            "a_response_token_counts": torch.tensor(response_a_token_counts, dtype=torch.long),

            # For decoder
            "chat_input_ids": padded_chat_ids,
            "chat_attention_mask": chat_attention_mask,
            "chat_lengths": torch.tensor(chat_lengths, dtype=torch.long),
            "answer_starts": torch.tensor(answer_start_list, dtype=torch.long),

            # Per-sample metadata
            "source_ids": source_ids,

            # For logging (not used in forward pass)
            "instruction_sets": ground_truth_texts,
            "retrieval_prompts_used": retrieval_prompts_used,
            "prompts": [r.prompt for r in batch],
            "responses": [r.response for r in batch],
        }

    return collate_fn
