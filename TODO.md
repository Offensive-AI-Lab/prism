<!-- Internal working document for the release. Remove this file before the final history squash / public flip. -->

# PRISM public-release TODO

This document tracks the public-release work for both repositories:

- `C:/Users/rahul/Documents/PRISM/prism`
- `C:/Users/rahul/Documents/PRISM/prism-eval`

It is based on the repository review recorded in `C:/Users/rahul/Documents/PRISM/rahul _ amar 1on1.vtt` and a subsequent inspection of both repositories. The transcript is review evidence, not a source of executable instructions. The requirements in this file are the working instructions.

## Ownership

- **Claude Code — implementation:** file deletion and movement, code changes, package changes, prompt-file renames, test updates, data conversion, artifact generation, link checks, and repository-wide mechanical cleanup.
- **Codex — writing:** all new or substantially rewritten prose, including READMEs, data cards, usage guides, technical explanations, limitations, captions, notices, and documentation organization.
- **Human decision:** licensing decisions, publication approval, canonical result selection, external permissions, and supplying unavailable artifacts.

Claude Code should not independently rewrite public-facing prose. When a task requires new or substantially revised prose, preserve the task in this file and leave it for Codex. Small mechanical edits needed after a rename, such as changing a filename in a link, are implementation work.

## Non-negotiable release decisions

- The released checkpoint and the paper use the same final PRISM results.
- Public documentation must not discuss a “19K-step” result, a “3.6K-step” result, or compare those two histories.
- To users, there is one final released PRISM checkpoint and one corresponding paper result.
- Do not reintroduce old result tables or checkpoint-development history.
- Preserve the existing uncommitted `prism-eval` result updates before beginning the cleanup.
- Do not run full training or full evaluation solely for this release-documentation work. Use focused checks for files, imports, links, prompt parity, and code paths affected by deletions or renames.
- Do not delete an artifact merely because its purpose is unclear. First determine whether code, tests, paper reproduction, or a released result depends on it.
- Public prose should sound authored by the project team: direct, specific, technically accurate, and free of generic filler, excessive caveats, and internal lab narration.

## Status legend

- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete
- `[?]` Blocked on a human decision or unavailable artifact

---

## Codex handoff — implementation state as of 2026-09-02 (Claude Code)

Everything mechanical in P0/P1/P4 that does not require a human decision is done
(see the ticked boxes). What exists now, so the writing can reference it:

- **Local demo shipped** at `prism/demo/` (`uv sync --extra demo && uv run python
  demo/app.py` → http://127.0.0.1:7860). Auto-downloads and SHA-verifies the two
  released Qwen checkpoints; ~90 s cold start, chat ~5 s, retrieval 2–4 s,
  19.3 GB VRAM. Four showcase prompts ship in `demo/example_prompts.json`.
- **Compare mode shipped**: LatentQA + Activation Oracles run as extra LoRA
  adapters on the same Qwen3.5-9B instance (peak 20.0 GB for a three-way
  compare). Adapters live on HF as
  `Offensive-AI-Lab/prism-baseline-latentqa-qwen3.5-9b` (116 MB, prefiltered)
  and `Offensive-AI-Lab/prism-baseline-activation-oracles-qwen3.5-9b` (465 MB);
  the first compare click downloads (~580 MB) and verifies them.
  `PRISM_DEMO_DISABLE_COMPARE=1` hides the mode. Vendored baseline code is under
  `demo/third_party/{lit,nl_probes}` with upstream LICENSE files (Apache-2.0 /
  MIT; lit carries local Qwen3.5 modifications).

Pending writing (Codex), in suggested order:

1. `prism/README.md` rewrite per task 9 — now including the "Try PRISM"
   quickstart (before training), the demo screenshot placement, and one short
   note that compare mode exists and what it downloads.
2. **Model cards + license tags for the two new HF baseline-adapter repos**
   (currently file-only, no card, no license metadata) and for the four
   checkpoint repos if they lack cards.
3. Task 8 Oracle-terminology replacement, tasks 10–15 document rewrites,
   task 16 visuals, task 23 editorial pass — unchanged scope.
4. Demo UI text review (labels, error wording, settings hints in
   `demo/index.html`) per task 17's Codex ownership.

Pending humans: HF org/name + approval for the training dataset (task 2);
XPIA provenance path (task 4); licensing decisions (task 3); the two `[?]`
demo-behavior confirmations in task 18; final squash of both repos at the end.

### Update — 2026-09-07 (Claude Code): dataset + XPIA shipped

Two big items since the note above are now DONE in code; the prose they create is Codex's.

**Training dataset (task 2) — released, private for now.**
- Uploaded to Hugging Face as `Offensive-AI-Lab/prism-training-dataset` (**private** pending review): `if_eval.jsonl` / `if_multi_constraints.jsonl` / `ultrachat.jsonl` (277,496 records), `valid_record_ids.json` (203,589-id mask), `source_inventory.json`.
- Record fields were renamed repo-wide: `prompt_a→prompt`, `response_a→response`, `response_b→instruction_set`; the constant `prompt_b` is dropped (code-only). Legacy names still load via fallbacks.
- New tools: `scripts/download_dataset.py` (fetch + SHA-verify into `$PRISM_DATA_DIR/prompt-only/`) and `scripts/check_dataset.py` (fields/dupes/counts/mask/split). prism README, PIPELINE §1, RECIPES, DATA_CARD now lead with the released dataset.
- **Codex prose to do:** write the HF **dataset card** (facts in `source_inventory.json`; counts train 162,821 / val 20,410 / test 20,358; `response` is ≤2048 generated tokens, ~5–12% cap-truncated); after the public flip, add direct dataset links to both READMEs and the prism data card.
- **Human pending:** review + flip the dataset repo public; formal licensing sign-off (IFEval Apache-2.0 / IF_multi_constraints ODC-BY-attribution / ultrachat MIT; Qwen outputs Apache-2.0).

**XPIA corpus (task 4) — reconstructed, no longer shipped (eval repo).**
- `data/xpia_corpus.parquet` **and** the derived `data/eval_suite_xpia_smoke.json` are removed (both embedded upstream benchmark text). `scripts/build_xpia_corpus.py` rebuilds a comparable corpus at the exact released counts (25,002) from user-fetched BIPIA/InjecAgent + HF LLMail, following Fomin (arXiv:2602.14161) and `maxf-zn/prompt-mining` (MIT © Zenity/Z Labs).
- NOTICE / DATA_CARD / README provenance corrected and attributed; the "taxonomy layer is this project's" overclaim removed. RESULTS.md left untouched as the reported reference.
- The rebuild gives the **same benchmarks at the same volume, not identical rows** (original selection + fine LLM taxonomy were private); shipped tags are coarse/deterministic. This is stated in DATA_CARD.
- **Codex prose to do:** editorial polish of the eval XPIA blocks (NOTICE wording, DATA_CARD XPIA section, README XPIA note) to the direct human-authored style — the current text is minimal Claude-authored provenance, factually correct but not styled.

**HF model/dataset cards still owed by Codex:** the two baseline-adapter repos (`prism-baseline-latentqa-qwen3.5-9b`, `prism-baseline-activation-oracles-qwen3.5-9b`), the training-dataset repo, and the four checkpoint repos if bare — all currently file-only, no card, no license tag.

---

## P0 — Resolve release blockers

### 1. Protect the existing `prism-eval` result changes

**Owner: Claude Code**

- [x] Inspect the current `prism-eval` working-tree diff before making additional changes.
- [x] Preserve all 37 existing modified files from the approved final-results update.
- [x] Commit the final-results update separately before beginning broad repository cleanup, unless Rahul explicitly asks for a different commit structure.
- [x] Confirm that the final PRISM row is the only public Qwen GRPO result.
- [x] Search both repositories for public references to `19K`, `19000`, `3.6K`, `3600`, `best@19000`, and the older Qwen checkpoint lineage.
- [x] Remove user-facing step-count and old-checkpoint references.
- [x] Keep step values only where software or checkpoint metadata genuinely requires them; do not feature them in READMEs, results prose, or release narratives.
- [x] Ensure no cleanup operation overwrites the approved final numbers currently present in the uncommitted `prism-eval` changes.

### 2. Release the actual PRISM training dataset

**Owners: Claude Code for artifact preparation; Codex for the dataset card and public explanations; human approval for publication**

- [x] Locate the exact filtered JSONL records used to train the released checkpoints. *(Verified from every released run's embedded config; byte-identical across all three target-model caches.)*
- [x] Confirm whether every target-model variant used the same filtered record set. *(Yes — hash-identical files and mask for all three GRPO runs.)*
- [x] Identify and retain only records accepted by the released `valid_record_ids.json` mask. *(Shipped as full set + mask: the split is computed before masking, so a masked-only release would change train/val membership.)*
- [x] Preserve stable record IDs and paraphrase-group IDs needed to reproduce split membership.
- [x] Prepare a Hugging Face-compatible dataset containing, at minimum: *(Fields renamed: prompt / response / instruction_set; the constant retrieval prompt is code-only.)*
  - [x] stable record ID;
  - [x] source dataset;
  - [x] `prompt_a`; *(now `prompt`)*
  - [x] `response_a`; *(now `response`, ≤2048 generated tokens — 5–12 % per source hit the cap)*
  - [x] `prompt_b`; *(dropped — one constant string across all 277,496 records; lives in code)*
  - [x] generated instruction list in `response_b`; *(now `instruction_set`)*
  - [x] split or grouping information needed to reproduce training; *(paraphrase_group_id + deterministic seed-42 split)*
  - [x] relevant generation metadata;
  - [x] relevant filtering metadata. *(the mask file)*
- [x] Record exact counts by source and split. *(277,496 total; masked 203,589; train 162,821 / val 20,410 / test 20,358; per-source table in check_dataset.py.)*
- [x] Add a deterministic validation command that checks required fields, duplicate IDs, split integrity, and record counts before upload. *(scripts/check_dataset.py, pinned val-membership checksum.)*
- [x] Decide the Hugging Face organization and dataset repository name. *(Offensive-AI-Lab/prism-training-dataset.)*
- [~] Publish the approved dataset to Hugging Face. *(Uploaded PRIVATE per Rahul; flip public after review + Codex dataset card.)*
- [x] Make the released dataset the primary/default path in reproduction documentation. *(PIPELINE §1, RECIPES, README sentence, recipe error message; Codex polish pending.)*
- [x] Retain the generation scripts as an optional way to create a new dataset, not as the only way to train PRISM.
- [~] Add direct dataset links from both repository READMEs and the `prism` data card after publication. *(prism README/data card done; eval README + Codex pass pending the public flip.)*
- [x] Clearly distinguish exact reproduction using the released records from generating a new sampled dataset.

### 3. Complete a source-by-source licensing audit

**Owners: human/legal decision with Claude Code collecting metadata; Codex writing the final licensing explanation**

- [x] Verify the exact dataset card, license, and redistribution terms for `google/IFEval`. *(apache-2.0, rev 966cd89545d6.)*
- [x] Verify the exact dataset card, license, and redistribution terms for `allenai/IF_multi_constraints_upto5`. *(odc-by — attribution required, rev 2e3a77407b7f.)*
- [x] Verify the exact dataset card, license, and redistribution terms for `HuggingFaceH4/ultrachat_200k`. *(mit, rev 8049631c405a.)*
- [x] Verify the terms governing the target model's generated `response_a` and `response_b` outputs. *(Qwen3.5 is Apache-2.0; outputs unencumbered.)*
- [x] Record the exact upstream dataset revision or commit where practical. *(In source_inventory.json, shipped with the dataset.)*
- [ ] Determine separately whether prompts, generated responses, and derived labels may be redistributed.
- [ ] Exclude any source or field that cannot legally be redistributed.
- [ ] If a source must be excluded, document which source is absent and provide a regeneration path for users with lawful access.
- [x] Produce a machine-readable source inventory with source name, upstream URL, revision, license, included fields, transformations, and redistribution decision. *(source_inventory.json.)*
- [ ] Create a clear Hugging Face dataset card covering sources, licenses, transformations, filtering, splits, intended use, and limitations.
- [ ] Do not imply that being hosted on Hugging Face automatically grants redistribution rights.
- [ ] Ensure `LICENSE`, `NOTICE`, repository data cards, and the Hugging Face dataset card agree.

### 4. Resolve the provenance of `prism-eval/data/xpia_corpus.parquet`

**Owners: Claude Code for forensic comparison and reconstruction; human approval/permission; Codex for final provenance documentation**

- [?] Locate the original Parquet or source artifact received from Microsoft or Zenity.
- [ ] Record a checksum for the original artifact and the repository copy.
- [ ] Determine whether the repository file is byte-identical to the received artifact or was subsequently modified.
- [x] Search for an official public Zenity release of the same assembled dataset. *(Found: M. Fomin, "When Benchmarks Lie", arXiv:2602.14161 + github.com/maxf-zn/prompt-mining, MIT © Zenity/Z Labs — the public loaders/paper behind the corpus.)*
- [x] Compare schema, row counts, IDs, content, labels, and taxonomy columns with the public BIPIA, LLMail, and InjecAgent sources. *(Counts explained: injecagent 1,054 = base cases; llmail 9,998 = ~10K cap on Phase1; bipia 13,950 = 13,750 attack + 200 benign, a private trim of the paper's 15K. Fine taxonomy = a private LLM pass, not in the public repo.)*
- [ ] Determine exactly who created or transformed:
  - [ ] normalized `content`;
  - [ ] dataset identifiers;
  - [ ] attack/benign labels;
  - [ ] `attacker_goal`;
  - [ ] delivery and evasion technique fields;
  - [ ] injection position;
  - [ ] tool-output type;
  - [ ] scope;
  - [ ] taxonomy source and rationale;
  - [ ] tier categories;
  - [ ] the 200 benign rows;
  - [ ] filtering, deduplication, or error-row handling.
- [?] Ask Julia, Microsoft, or Zenity for redistribution confirmation if the assembled artifact is not publicly documented.
- [?] Select one defensible release path:
  1. attribute and redistribute a confirmed public Zenity artifact;
  2. redistribute with explicit written permission and full provenance;
  3. rebuild the corpus in this repository from the three public upstream sources; **← chosen (A2)**
  4. remove the Parquet corpus and associated XPIA release claims until provenance is resolved.
- [x] If rebuilding, create a deterministic script that fetches pinned upstream revisions and produces the documented schema. *(scripts/build_xpia_corpus.py, seed 2024, exact per-source counts, same 14-column schema.)*
- [x] If rebuilding, compare counts and representative rows against the current corpus and explain any result-affecting differences. *(Counts match exactly (25,002); rows are not identical (different selection) — documented in DATA_CARD; RESULTS.md stays as the reported reference.)*
- [x] Remove or correct the current unsupported claim that the taxonomy/tagging layer was created entirely by this project.
- [x] Update `prism-eval/NOTICE`, `prism-eval/DATA_CARD.md`, `prism-eval/README.md`, and XPIA result documentation to match the verified origin. *(RESULTS.md left as the reported reference per Rahul.)*
- [x] Do not present the XPIA corpus as release-ready while permission or provenance remains unresolved.

### 5. Establish one source of truth for calibration results

**Owners: human decision; Claude Code for artifact cleanup and reproducibility checks; Codex for final explanation**

- [x] Confirm whether the current paper still reports judge-vs-gold quadratic-weighted kappa as `0.817`. *(Yes — Rahul: repo docs cite the paper's 0.817 / 0.823.)*
- [?] Locate the exact saved artifact behind `0.817`, if it exists.
- [x] If that artifact cannot be recovered, decide whether the paper and repository should use the reproducible result of approximately `0.800`. *(Decision: docs cite the paper's Table 4 values; the shipped artifacts/scripts compute ≈0.800/0.824 from the raw labels.)*
- [x] Select one canonical calibration artifact for the scoring judge.
- [ ] Select one canonical calibration artifact for the adversarial-instruction identifier.
- [ ] Select one canonical calibration artifact for the optional FOLLOWED/behavior judge, if that analysis remains public.
- [~] Ensure every published calibration number can be reproduced from a shipped artifact. *(Docs cite the paper's 0.817/0.823 per Rahul; the shipped labels + calibrate_judge.py compute ≈0.800/0.824.)*
- [x] Remove prose that narrates several unsuccessful attempts to reconstruct the paper number.
- [ ] Keep the checkpoint-results issue separate from the calibration-table issue: the final released checkpoint and main paper result already match.

---

## P1 — Remove internal development history

### 6. Clean the `prism` repository

#### 6.1 Replace the standalone known-issues document

**Owners: Claude Code for classification and code fixes; Codex for any retained limitation prose**

- [x] Review every item in `prism/docs/KNOWN_ISSUES.md`.
- [x] Classify each item as one of:
  - [ ] a real current bug that should be fixed;
  - [ ] a user-facing limitation that belongs next to the relevant feature;
  - [ ] a historical implementation detail that should be deleted;
  - [ ] a paper limitation that belongs in the paper or a concise limitations section.
- [x] Fix current bugs where a safe correction does not invalidate released checkpoints or reproduction. *(None were current bugs; retained-behavior items were deliberate.)*
- [x] Move genuinely necessary limitations into the relevant pipeline, data, or configuration section.
- [x] Delete historical and defensive narration.
- [x] Remove `docs/KNOWN_ISSUES.md` after all retained information has a proper home.
- [x] Remove links to `docs/KNOWN_ISSUES.md` from the README and other documents.

#### 6.2 Consolidate calibration ownership in `prism-eval`

**Owner: Claude Code, with Codex rewriting any affected documentation**

- [x] Confirm that no production training path imports `prism.calibration`.
- [x] Compare `src/prism/calibration/` with the calibration code in `prism-eval`.
- [x] Identify any unique operation that is still needed for paper reproduction. *(None — prism-eval's calibrate_judge.py + shipped gold covers it; rederive_rewards moved to prism.rl as a reward-weight tool.)*
- [x] Move genuinely needed unique functionality into `prism-eval` with an appropriate public interface. *(Nothing needed moving.)*
- [x] Remove `src/prism/calibration/` from `prism` once dependencies are resolved.
- [x] Remove the `calibration` optional dependency group from `prism/pyproject.toml` if no longer needed.
- [x] Remove calibration references from `src/prism/__init__.py` and package documentation.
- [x] Delete `prism/docs/CALIBRATION.md` after relevant eval-side material is consolidated.
- [x] Remove the judge-calibration section from `prism/docs/PIPELINE.md`.
- [x] Remove judge-calibration material from the `prism` data card.

#### 6.3 Remove historical rubric variants from `prism`

**Owner: Claude Code, with Codex updating rubric prose and links**

- [x] Delete `docs/judge_rubrics/hallucination_rubric_baseline.txt`.
- [x] Delete `docs/judge_rubrics/hallucination_rubric_expanded.txt`.
- [x] Delete `docs/judge_rubrics/hallucination_rubric_released.txt` after confirming the canonical prompt is preserved elsewhere.
- [x] Delete `docs/judge_rubrics/hallucination_rubric_three_source.txt`.
- [x] Remove the empty `docs/judge_rubrics/` directory.
- [x] Keep one canonical scoring rubric in `prism-eval`.
- [x] Link to that canonical rubric from the GRPO documentation.
- [x] Confirm that the GRPO reward prompt and eval scoring prompt remain equivalent after cleanup.
- [x] Retain or update an automated parity check where practical.

#### 6.4 Simplify recipes and training documentation

**Owners: Claude Code for factual/configuration audit; Codex for the rewrite**

- [x] Remove the obsolete “second Qwen GRPO system” and judge-swap reproduction recipe from `docs/RECIPES.md`.
- [x] Remove old public references to `best@19000`, `best@9400`, and other checkpoint-development history.
- [x] Retain the actual released hyperparameters needed for reproduction.
- [x] Explain that recipe scripts, rather than module defaults, define the released training configuration. *(README + RECIPES both state it.)*
- [x] Remove internal validation scores and selection narration unless required to choose the correct released artifact.
- [x] Remove tensor-identical/export-forensics claims that do not help users reproduce or use PRISM.
- [x] Review `docs/CHECKPOINT_FORMAT.md` and remove unnecessary historical checkpoint-schema commentary.
- [x] Preserve technical fields that loaders genuinely require.
- [x] Review `docs/ABLATIONS.md` and decide whether it describes a completed, reproducible public experiment or an incomplete internal plan. *(Keep: the layer sweep and seed study are paper-appendix experiments; commands and inputs are public.)*
- [x] Keep the ablation document only if its commands, inputs, outputs, and corresponding results are all public and usable.

#### 6.5 Audit `prism` scripts and utilities

**Owner: Claude Code**

- [x] Retain the dataset generation and cleaning scripts if they can produce a documented new dataset.
- [x] Retain activation extraction and on-the-fly parity tools if they support documented public workflows.
- [x] Retain checkpoint export and chat-template validation tools.
- [x] Audit `scripts/analyze_judge_traces.py`; retain only if it supports a documented GRPO debugging or reproduction workflow. *(Retained: documented in PIPELINE §4.)*
- [x] Audit `prism.rl.build_hard_ids`; retain only if the hard-example curriculum remains a supported public feature. *(Retained: public CLI + config + PIPELINE mention.)*
- [x] Remove references to tools that are retained solely for internal experiment management.

### 7. Clean the `prism-eval` repository

#### 7.1 Remove judge-selection experiments

**Owner: Claude Code**

- [x] Delete `data/calibration/judge_swap/gemma4-31B-it/`.
- [x] Delete `data/calibration/judge_swap/gpt-oss-120b/`.
- [x] Delete `data/calibration/judge_swap/seed-oss-36b/`.
- [x] Remove the empty `data/calibration/judge_swap/` directory.
- [x] Remove all references to judge-swap dev/holdout reports from data cards, result documents, tests, and scripts.
- [ ] Remove public discussion of judge-selection experiments that are not part of the paper's reproducible release.

#### 7.2 Replace calibration experiment traces with final artifacts

**Owners: Claude Code for artifact selection/removal; Codex for rewritten calibration documentation**

- [x] Remove `data/calibration/published/judge_vs_gold_run1.json` after the canonical result decision.
- [x] Remove `data/calibration/published/judge_vs_gold_run2.json` after the canonical result decision.
- [x] Remove `data/calibration/published/judge_vs_each_annotator.json` unless it is required to reproduce a paper table.
- [x] Retain `inter_annotator_pilot.json` only if it is the canonical artifact required for a reported result and cannot be cleanly represented by the versioned top-level calibration report. *(Removed: the top-level report carries human-vs-human 0.8239/170 recomputed from the raw labels.)*
- [x] Replace the `published/` directory with clearly named canonical artifacts, or remove the directory if the top-level `*_v1.json` reports are sufficient.
- [x] Update tests to validate canonical artifacts rather than historical runs.
- [x] Audit calibration JSONL files for internal call IDs, timestamps, account fields, paths, or unnecessary metadata. *(Already pseudonymized — annotator_name/wb_user_id hold only annotator_a/b; call_id is the functional pairing key; a test guards this.)*
- [x] Preserve pseudonymized human labels and fields required to recompute the reported statistics.

#### 7.3 Keep only final judge prompts

**Owner: Claude Code, with Codex reviewing names and explanatory text**

- [x] Delete `prism_eval/prompts/judge/variants/v1_baseline.txt`.
- [x] Delete all `v2_*` files under `prism_eval/prompts/judge/variants/`.
- [x] Remove the empty variants directory.
- [x] Keep only the final scoring-judge and adversarial-identifier prompts.
- [x] Rename `published_judge.txt` to a functional name such as `scoring.txt`.
- [x] Rename `published_advdet.txt` to a functional name such as `adversarial_identifier.txt`.
- [x] Update `tests/test_judge_prompt_parity.py` and all rubric links after renaming.
- [x] Remove prompt-development and variant history from `RUBRIC.md`.
- [ ] Consider loading the canonical prompt files from code instead of duplicating large strings, provided packaging remains reliable.
- [x] If prompts remain embedded in code, preserve a strict parity test against the canonical files.

#### 7.4 Remove internal review tools and audit scripts

**Owner: Claude Code**

- [x] Delete `scripts/view_xpia_suite.py`; it is an internal HTML review helper.
- [x] Retain `scripts/analyze_xpia.py`; it computes the XPIA tables and breakdowns used in `docs/RESULTS.md`.
- [x] Remove the “ported unchanged from internal reports” commentary from `scripts/analyze_xpia.py`.
- [ ] Confirm that `scripts/analyze_xpia.py` works without optional judge calls for its structural analysis.
- [ ] Clearly mark behavior/provenance judge calls as optional if those analyses remain public.
- [x] Audit `scripts/strip_checkpoint.py`; checkpoint export and release preparation may belong only in `prism`. *(Audited: retained in prism-eval — it is a generic inference-stripping utility referenced by `download_weights.py`; prism's exporter covers release preparation.)*
- [~] Audit `scripts/label_injection_success.py` and retain it only if it is part of a reproducible suite-building path. *(Retained pending the XPIA provenance decision; internal plan narration removed.)*
- [x] Retain `scripts/serve_judge.sh`; it is a generic and useful vLLM launch script.
- [ ] Audit every remaining script against one criterion: it must support runtime use, paper reproduction, public data construction, or a documented extension workflow.

#### 7.5 Audit annotation queue artifacts

**Owners: Claude Code for dependency analysis; Codex for any retained annotation guide**

- [x] Determine whether `configs/advdet_queue_spec_v1.json` is needed to reproduce a reported calibration artifact. *(Retained: documents the shipped advdet gold set's sampling; linked from DATA_CARD.)*
- [x] Determine whether `configs/follow_queue_spec_v1.json` is needed to reproduce a reported calibration artifact. *(Retained: referenced by docs/ANNOTATION_FOLLOW.md; snapshot + report are shipped.)*
- [x] Determine whether `configs/recall_calib_spec_v1.json` is needed to reproduce a reported calibration artifact. *(Deleted: internal Weave-trace experiment, unreferenced.)*
- [x] Retain queue specifications only if their inputs, sampling method, labels, and outputs are sufficiently public to reproduce the process.
- [x] Audit `docs/ANNOTATION_FOLLOW.md` using the same criterion. *(Retained; internal paths and dangling doc references fixed. Codex rewrite still pending per task 15.)*
- [ ] If retained, consolidate annotation instructions and queue provenance into one concise calibration document.
- [ ] If not retained, remove the specs, document, and dangling references.

---

## P2 — Rewrite the public documentation

All tasks in this section are owned by **Codex**, except factual verification, mechanical link updates, and asset placement, which Claude Code may perform.

### 8. Replace “Oracle” in the public interface

- [ ] Use “instruction-set generation” for the pipeline stage that generates instruction labels.
- [ ] Use “training-data generation” when referring to the broader process of building examples.
- [ ] Use “generated instruction labels” or “instruction reports” for `response_b`, depending on context.
- [ ] Replace “oracle reports,” “oracle data,” and “Oracle data generation” in both READMEs.
- [ ] Replace the term in public data cards, pipeline guides, recipe guides, comments shown in examples, and CLI help.
- [ ] Avoid renaming unrelated citations such as the external method “Activation Oracles.”
- [?] Decide whether internal Python identifiers such as `oracle_mode`, `ORACLE_MODES`, output filenames, and historical checkpoint fields should also be renamed.
- [ ] If code-level identifiers are renamed, preserve compatibility aliases for existing scripts, data, and checkpoints.

### 9. Rewrite `prism/README.md` around user intent

Use this information order:

1. One-paragraph explanation of PRISM.
2. A “Try PRISM” quickstart using the released checkpoint.
3. A simple “Why PRISM?” visual.
4. Direct checkpoint and dataset links.
5. A short “How it works” diagram and explanation.
6. Training prerequisites.
7. The released training path.
8. Target-model extension notes.
9. Citation and license.

Tasks:

- [ ] Rewrite the opening for a reader who has not read the paper.
- [ ] Explain what PRISM consumes and what it produces in concrete terms.
- [ ] Remove the cryptic paragraph saying that training data are generated locally and not distributed.
- [ ] Lead with using the released model rather than training from scratch.
- [ ] Reduce the pipeline to three conceptual stages:
  1. create instruction-labelled examples;
  2. read response-token activations and train the decoder;
  3. refine with GRPO and export.
- [ ] Explain precomputed versus on-the-fly activation extraction in one short note.
- [ ] State that precomputation is the released high-throughput reproduction path and on-the-fly extraction is supported but optional.
- [ ] Avoid implying that activation precomputation is inherently mandatory for GRPO.
- [ ] Move caching, split parity, model-class differences, and detailed extraction behavior into the pipeline guide.
- [ ] Remove the long reproduction-caveat list from the README.
- [ ] Add direct Hugging Face links beside every released checkpoint.
- [ ] Add the released training-dataset link beside the training instructions.
- [ ] State clearly which checkpoint is the primary paper model without mentioning historical step counts.
- [ ] Keep links to deeper documents selective; do not recreate a large contents/index section.

### 10. Rewrite `prism/docs/DATA_CARD.md` from scratch

- [ ] Begin with the exact upstream sources and their licenses.
- [ ] State what data are released and where to download them.
- [ ] Show one representative record or compact schema example.
- [ ] Explain `prompt_a` as the instruction-rich user request.
- [ ] Explain `response_a` as the frozen target model's response.
- [ ] Explain `prompt_b` as the fixed request for an instruction report.
- [ ] Explain `response_b` as the generated instruction list used as the training target.
- [ ] Explain which fields SFT consumes.
- [ ] Explain which fields GRPO consumes.
- [ ] Explain the validity mask, filtering criteria, grouping, and split behavior.
- [ ] Report released source and split counts.
- [ ] Link the released Hugging Face dataset.
- [ ] Explain how regenerating data differs from using the released records.
- [ ] Remove judge-calibration content; it belongs in `prism-eval`.
- [ ] Remove rubric-version history.
- [ ] Remove references to the deleted known-issues document.
- [ ] Keep licensing language specific, accurate, and readable.
- [ ] Avoid phrases that merely announce what the document “ships,” “covers,” or “documents” without conveying substantive information.

### 11. Rewrite `prism/docs/PIPELINE.md`

- [ ] Align section names and stage numbering with the README architecture diagram.
- [ ] Replace public “Oracle” terminology.
- [ ] Describe the released dataset path first and generation of a new dataset second.
- [ ] Explain activation precomputation and on-the-fly extraction accurately.
- [ ] State that on-the-fly overhead should be the additional no-grad activation-extraction forward for each sampled batch, assuming implementation avoids unrelated duplicated work.
- [ ] Keep details about split identity, valid-record masks, hook layers, and cache reuse here rather than in the README.
- [ ] Remove calibration ownership and rubric-history material.
- [ ] Remove internal experiment commentary such as optional curricula unless the feature remains publicly supported.
- [ ] Use one consistent name for the target model, PRISM decoder, judge model, report, and instruction labels.

### 12. Rewrite and simplify the remaining `prism` documents

- [ ] Rewrite `docs/RECIPES.md` as a concise released-training reference after Claude Code removes historical material.
- [ ] Rewrite `docs/CHECKPOINT_FORMAT.md` to describe only current training and release formats plus the security note.
- [ ] Rewrite or remove `docs/ABLATIONS.md` based on the completed-public-experiment decision.
- [ ] Review `docs/RUBRIC.md`; replace it with a short link to the canonical eval rubric if maintaining two copies creates drift.
- [ ] Review `CONTRIBUTING.md` for the same direct, human-authored style.
- [ ] Ensure no document refers readers to deleted files or internal-only artifacts.

### 13. Rewrite and tighten `prism-eval/README.md`

- [ ] Keep the opening understandable without the paper.
- [ ] Explain that evaluation involves two distinct model roles:
  - [ ] the local target model plus PRISM checkpoint;
  - [ ] the judge model exposed through an OpenAI-compatible endpoint for paper metrics.
- [ ] State which quickstart requires only the target model and which workflows require the judge.
- [ ] Clarify whether the 80 GB figure refers to PRISM inference, a co-located judge, or the combined setup.
- [ ] Keep the existing smoke evaluation as a useful installation check.
- [ ] Do not present the smoke evaluation as the interactive “Try PRISM” demo.
- [ ] Delete “Exact match and token F1 are also emitted, but they are not the paper's headline metrics.”
- [ ] Remove the equivalent implementation-check sentence from `docs/REPRODUCING.md` unless it is genuinely needed there.
- [ ] Keep the metrics table focused on reported metrics.
- [ ] Add direct Hugging Face links to all released checkpoints.
- [ ] Remove redundant explanations duplicated in `docs/REPRODUCING.md`.
- [ ] Keep XPIA out of the public release narrative until provenance is resolved.
- [ ] If XPIA remains, summarize it briefly and point to one authoritative provenance section.
- [ ] Ensure the result discussion contains only final paper/released-checkpoint results, with no historical step labels.
- [ ] Keep the supported `evaluate` path prominent and present lower-level commands only if they are useful to external users.
- [ ] Remove generic sentences and unnecessary warnings that do not help a reader act.

### 14. Rewrite `prism-eval/DATA_CARD.md`

- [ ] Separate the canonical 1,000-record evaluation suite from the optional XPIA corpus.
- [ ] Put source, license, record count, and transformation information in a compact table.
- [ ] Explain AP, HO, BC, and BN directly and without promotional filler.
- [ ] Retain the non-commercial BN restriction prominently and accurately.
- [ ] Retain relevant content limitations for BC without listing colorful examples unless necessary.
- [ ] Explain the suite schema with one representative record.
- [ ] Describe AP/HO construction and filtering concisely.
- [ ] Replace the current calibration narrative with the final canonical artifacts and reproducible numbers only.
- [ ] Remove third-rater narration, failed-run history, “worth knowing” commentary, and judge-selection history.
- [ ] Document XPIA only after its provenance path is resolved.
- [ ] Ensure all provenance and licensing claims match `NOTICE` and the actual released files.
- [ ] Retain genuine limitations: single-response evaluation, primarily English data, synthetic adversarial scenarios, and target-model dependence.

### 15. Rewrite remaining `prism-eval` documentation

- [ ] Rewrite `docs/RESULTS.md` so it reports the final paper results and directly reproducible supplemental results only.
- [ ] Remove old checkpoint comparisons and step histories.
- [x] Remove the calibration discrepancy narrative once a canonical decision is made.
- [ ] Retain upstream baseline revisions and checkpoint links because they are useful for provenance.
- [ ] Rewrite `docs/REPRODUCING.md` as the detailed counterpart to the concise README.
- [ ] Make target-model, PRISM-checkpoint, and judge-endpoint requirements explicit.
- [ ] Keep realistic sources of run-to-run variation without overexplaining.
- [ ] Review `docs/ABLATION_REPORT.md` for internal narration, stale checkpoint names, and consistency with final results.
- [ ] Review `RUBRIC.md` and `RUBRIC_ADVDET.md` for concise public-facing terminology and canonical prompt links.
- [ ] Rewrite or remove `docs/ANNOTATION_FOLLOW.md` according to the annotation-artifact decision.
- [ ] Review `CONTRIBUTING.md` for the same direct, human-authored style.

### 16. Add two useful visuals

**Owners: Codex for concept, captions, and visual direction; Claude Code for placing assets and updating links; human approval for style**

- [ ] Create a simple “Why PRISM?” teaser showing:
  - [ ] an agent performing a task;
  - [ ] the limitation of asking the agent to self-report what it is doing;
  - [ ] PRISM reading internal activations;
  - [ ] PRISM returning the instructions currently represented.
- [ ] Create or adapt a clean PRISM architecture/pipeline diagram.
- [ ] Label its stages to correspond exactly to the README and pipeline guide.
- [ ] Prefer repository-owned SVG source plus a rendered PNG only if GitHub compatibility requires it.
- [ ] Add useful alt text and concise captions.
- [ ] Reuse a consistent visual language in a future paper website.
- [ ] Avoid decorative AI imagery that makes the project appear less technical or authoritative.
- [?] Locate the editable source for the paper's existing architecture figure.

---

## P3 — Provide a real “Try PRISM” path

### 17. Locate and package the existing demo

**Owners: Claude Code for implementation; Codex for UI text, quickstart prose, labels, error wording, and screenshot caption**

- [x] Locate the Modal-connected UI code discussed in the review transcript. *(demo_app/ in the internal repo.)*
- [x] Determine which repository should own the local demo. *(prism repo, `demo/`.)*
- [x] Remove the Modal dependency from the public demo.
- [x] Package the demo to run on a user's local CUDA GPU. *(`uv sync --extra demo && uv run python demo/app.py`; verified live: ~21 GB VRAM, chat 5 s, retrieval 2–4 s.)*
- [x] Make the default demo use the final released Qwen PRISM checkpoint.
- [x] Download the checkpoint automatically or provide one clear download command.
- [x] Reuse the checksum verification already present in `prism-eval`. *(Same SHA-256 digests; verified it rejects a stale local export.)*
- [x] Do not require an LLM judge endpoint for ordinary interactive use.
- [x] Allow a user to enter a prompt.
- [x] Generate or accept the target model's response. *(UI generates; the API also accepts a pasted `assistant_response`.)*
- [x] Extract the relevant response-token activations.
- [x] Decode and display PRISM's recovered instruction report.
- [x] Display prompt, model response, and PRISM report with clear labels.
- [x] Use predictable model-cache and checkpoint directories. *(HF cache + ./checkpoints / PRISM_DEMO_CHECKPOINT_DIR.)*
- [x] Provide one supported launch command.
- [x] Add actionable errors for missing CUDA, insufficient memory, missing gated-model access, and checkpoint/model mismatch. *(CUDA, VRAM, OOM, SHA mismatch; the base model is not gated.)*
- [ ] Add a screenshot directly below the quickstart.
- [ ] Consider a Colab only after the local path is stable.
- [ ] Put “Try PRISM” before instructions for training PRISM from scratch.

### 18. Define the minimum supported demo experience

**Owners: human product decision; Claude Code implementation; Codex writing**

- [?] Decide whether the demo generates the target response itself or also supports pasting an existing response. *(Current behavior: the UI generates; the API accepts a pasted `assistant_response`. Confirm or change.)*
- [?] Decide whether the first release is a browser UI, a CLI, or both. *(A browser UI is implemented and verified; a CLI would be additional work.)*
- [ ] State minimum and recommended GPU memory based on the actual model-loading path. **Codex** — measured facts: 19.3 GB PRISM-only, 20.0 GB peak with compare; recommend "24 GB" in prose.
- [x] Ensure the demo does not silently send prompts or outputs to external services. *(Only outbound traffic is the Hugging Face model/checkpoint download; verified.)*
- [ ] Explain that no judge is needed to inspect PRISM reports. **Codex** — true of the implementation.
- [x] Include one short example showing a benign instruction and an injected or hidden instruction. *(Four example cards ship in demo/example_prompts.json: hidden objective, prompt injection, behavioural constraint, benign control.)*

---

## P4 — Final repository audit

### 19. Classify every tracked file

**Owner: Claude Code, with Codex reviewing all retained public prose**

- [x] Review every tracked file in `prism`. *(Full classification done; one debris file — logging_setup.py — removed.)*
- [x] Review every tracked file in `prism-eval`. *(Full classification done; dead projection path, cli gaps, Weave test leak fixed.)*
- [x] Classify each as:
  - [ ] required runtime code;
  - [ ] paper reproduction material;
  - [ ] public data or calibration evidence;
  - [ ] user-facing documentation;
  - [ ] supported extension tooling;
  - [ ] internal development history to remove.
- [x] Remove files in the final category after confirming no dependencies.
- [x] Record uncertain files in this TODO rather than silently retaining unexplained debris. *(Uncertain, retained deliberately: `scripts/label_injection_success.py` + `prism_eval/scoring/injection_compliance.py` (XPIA-dependent, pending task 4); `scripts/prefill_response_cache.py` (Weave-only convenience, usage now documented); `scripts/strip_checkpoint.py` (generic utility); `configs/ablation/window/pos_last128_eos.yaml` (env overrides in its header, not driver-launched).)*

### 20. Remove internal residue

**Owner: Claude Code**

- [x] Search for local absolute paths. *(Only scrub-marker tuples and an elided example remain.)*
- [x] Search for W&B project names, run IDs, entity names, and internal account identifiers.
- [x] Search for private endpoints, credentials, tokens, and internal hostnames.
- [x] Search for experiment commentary such as “ported from internal,” “reviewer request,” “later,” “TODO,” “old,” “legacy,” and “published run 1/run 2.” *(Remaining “legacy” hits are correctness comments about compat guards.)*
- [x] Remove abandoned alternatives and development-only notes.
- [x] Retain meaningful code comments that explain non-obvious correctness constraints.
- [x] Audit calibration data for privacy-sensitive metadata while preserving reproducibility.

### 21. Cross-repository consistency audit

**Owners: Claude Code for automated comparison; Codex for terminology and prose consistency**

- [x] Verify checkpoint names match across scripts, configs, READMEs, and Hugging Face.
- [x] Verify SHA-256 digests match released files. *(Verified against HF during the release; re-verify live before the announcement.)*
- [x] Verify target model IDs match.
- [x] Verify hook layers match.
- [x] Verify projection dimensions and activation-window settings match.
- [ ] Verify the final result values match the paper.
- [x] Verify dataset names and record counts match.
- [x] Verify judge model names and endpoint examples match.
- [x] Verify reward definitions and metric names match. *(test_reward_equivalence pins it.)*
- [x] Verify the canonical scoring prompt and rubric match the GRPO reward implementation. *(SYSTEM_PROMPT == scoring.txt, byte-identical, 6054 chars.)*
- [x] Verify citation metadata and author ordering match.
- [x] Verify repository and paper URLs match. *(arXiv 2606.09563 everywhere; confirm the HF checkpoint repos are public before announcing.)*
- [ ] Ensure README, data card, `NOTICE`, and license claims do not contradict one another.

### 22. Documentation and packaging checks

**Owner: Claude Code**

- [x] Check every relative Markdown link. *(0 broken in either repo.)*
- [~] Check every external link. *(Inventoried and sane; live-fetch of arXiv/HF links still to do at announcement time.)*
- [x] Check every documented command against existing files and CLI options. *(Agent-verified; the calibrate_advdet crash, calibrate_follow data-card command, cli help drift, and template gaps all fixed.)*
- [x] Check that renamed prompt files are included in the built Python package. *(Verified in the built wheel.)*
- [x] Check imports after removing `prism.calibration` and other files.
- [x] Run focused unit tests affected by file deletions, prompt renames, and packaging changes. *(prism 144, prism-eval 203, green after every change.)*
- [x] Run a lightweight offline evaluation test if the eval code changes. *(test_offline_evaluation + test_weave_eval_cache in the suite; Weave now fully disabled in tests.)*
- [ ] Run a lightweight dataset schema validation if data artifacts change.
- [ ] Do not run full training or the full 1,000/25,000-record evaluation solely for documentation cleanup.
- [x] Confirm both Git working trees contain only intended release changes before committing.

### 23. Final human editorial pass

**Owner: Codex, followed by Rahul/Gilad approval**

- [ ] Read both READMEs from top to bottom as a new user who has not read the paper.
- [ ] Read every retained public document for generated-sounding prose.
- [ ] Remove redundant summaries, scene-setting, “this document covers” language, and excessive warnings.
- [ ] Remove sentences that do not help a user understand, use, reproduce, extend, or cite PRISM.
- [ ] Check that technical terms are introduced before they are abbreviated.
- [ ] Keep detailed material in the appropriate linked document rather than repeating it in the README.
- [ ] Confirm that the final tone is confident but does not overclaim.
- [ ] Obtain final author approval for results, data provenance, licensing, visuals, and the demo instructions.

---

## P5 — Pre-release validation (added 2026-09-07; Claude Code drives)

Rigorous end-to-end smoke testing on real GPU + judge LLM + live artifact pulls
before the public flip, because the release touched load-bearing paths (the
`prompt_a→prompt` field rename across the training pipeline, on-the-fly GRPO,
the dataset release, the XPIA reconstruction, prompt renames, calibration
consolidation).

**Methodology — "no-clue" agents.** Each tier is run by a FRESH-CONTEXT agent
given only a clean clone and the repo's own README/docs — no insider knowledge.
Where the agent gets stuck is where a real user/tester gets stuck: those spots
are friction to fix (missing dep, unclear command, wrong path, absent env var),
not just test failures. Test from clones of the PUBLIC repos wherever possible.

**Resources:** up to 6 GPUs; judge LLM via `scripts/serve_judge.sh`
(gemma-4-31B-it); HF checkpoints/adapters (public) + dataset (private → token
until flipped). As of 2026-09-07: local GPU free; no judge live; dataset repo
private; `prism-eval` GitHub public but `prism` still private (its README link
to prism 404s for outsiders — a final-gate item).

### Tier 0 — offline / CPU
- [x] 0.1 `uv sync --extra dev` (+ demo, bertscore) on clean clones, both repos — resolves cleanly *(all extras exit 0; eval no longer forces cu128 torch)*
- [x] 0.2 `uv run pytest -q` both repos — prism 144 / eval 203, 0 fail, 0 skip
- [x] 0.3 wheel build; renamed prompts (`scoring.txt` 6074B, `adversarial_identifier.txt` 3446B) packaged
- [x] 0.4 every documented command + relative link in READMEs/docs resolves *(0 broken, incl. Codex's rewrites)*
- [x] 0.5 import every module from outside the repo — 32/32 prism, 22/22 eval; no module needs an extra just to import

**Tier 0 outcome (2026-09-07):** all pass, zero correctness failures. Newcomer friction found & fixed: (1) READMEs never documented the test command → added a Development section to both; (2) eval forced a CUDA torch build on CPU boxes → dropped the cu128 index (now consistent with prism); (3) a stale `requests`/urllib3 import warning on every run → pinned `requests>=2.32`. Remaining ~719 pytest warnings are Pydantic-deprecation noise from the `weave` dependency (not our code), left as-is. Pushed eval ee5dda5, prism c64d2e9.

### Tier 1 — artifact pulls (network, no GPU)
- [x] 1.1 dataset download → check_dataset = "matches the release exactly" (277,496; 162,821/20,410/20,358) *(added --token/$HF_TOKEN passthrough; anonymous once public)*
- [x] 1.2 checkpoint download + SHA — `prism-qwen3.5-9b-grpo` fetched anonymously, checksum OK (no token needed)
- [x] 1.3 baseline adapters — 4/4 files SHA-match; added standalone `scripts/download_baselines.py` (no-GPU fetch+verify, was demo/GPU-only)
- [x] 1.4 XPIA rebuild from clean clones → 25,002 (bipia 13,750+200, llmail 9,998, injecagent 1,054); noted BIPIA is clone-only (do not pip install)

**Tier 1 outcome (2026-09-07):** all pass; the public/private access model verified (dataset 401 private, checkpoints+adapters anonymous). Friction fixed: (1) dataset script had no token path → added `--token`/$HF_TOKEN + docstring note; (2) baseline adapters were fetchable only via the GPU demo path → added `scripts/download_baselines.py`; (3) BIPIA clone-vs-install ambiguity → header note. **For Codex:** the prism README data step should mention HF_TOKEN while the dataset repo is private, repeat the `export PRISM_DATA_DIR=...` in that section, and the demo README can point at `download_baselines.py`. Pushed prism a6ae88e, eval 4a961e6.

### Tier 2 — GPU, no judge
- [x] 2.1 SFT smoke, cache path — SFT DONE, val loss 2.681 (trained from the downloaded released dataset)
- [x] 2.2 SFT smoke, on-the-fly — SFT DONE, val loss 2.655
- [x] 2.3 SFT from the downloaded dataset — covered by 2.1 (PRISM_DATA_DIR = the HF download; renamed fields load fine)
- [x] 2.4 layer-ablation smoke — ABLATION RUN DONE, val loss 2.694
- [x] 2.5 demo PRISM-only — recovers the hidden Fast-Track objective; 35 s cold-start, ~19.3 GB VRAM
- [x] 2.6 demo compare — all three columns answer; adapters auto-downloaded + SHA-verified; ~20 GB peak

**Tier 2 outcome (2026-09-07):** all pass, zero code fixes. The field rename did not break any training path — cache, on-the-fly, and ablation all SFT-DONE with sane val loss on the freshly-downloaded released dataset; the demo works end to end (PRISM-only + 3-way compare) with correct hidden-objective recovery at ~20 GB. **For Codex (demo README):** document `--port` / `--host` / `--checkpoint-dir`; state that first launch needs network + HF but no login for the public repos; note `uv sync --extra demo` is self-contained (pass `--extra demo --extra dev` for both) — the recurring non-additive-extra gotcha (also Tier 0); optionally a brief note that a cached base model makes cold-start ~35 s vs the "~10 min" first-download estimate.

### Tier 3 — GPU + judge
- [x] 3.1 GRPO smoke, cache — RL DONE, 50 steps, 0 judge errors
- [x] 3.2 GRPO smoke, on-the-fly — RL DONE, 50 steps, 0 judge errors
- [x] 3.3 eval end-to-end with judge — coverage + adversarial-detection metrics written (renamed scoring.txt/adversarial_identifier.txt parse)
- [x] 3.4 XPIA end-to-end — build corpus (25,002) → judge-extract 8-record suite (3 sources) → judge-scored eval; summary written
- [x] 3.5 calibration — calibrate_judge reproduces 0.8002/0.8239; calibrate_advdet 49/50 (schema fix holds)

**Tier 3 outcome (2026-09-07):** all pass with a live gemma-4-31B-it judge; no code fixes. GRPO (both paths) runs judge-scored with 0 judge errors after the rename; the eval scoring path and the whole XPIA pipeline work end to end; calibration reproduces the shipped numbers. **Infra note (not release-blocking):** the internal `~/itm-eval-suite/vllm-service/serve_gemma.sh` used `uv run vllm` which failed to find the binary on a compute node — I launched via `~/.venv-vllm/bin/vllm` directly. The PUBLIC path is `scripts/serve_judge.sh` (creates its own `~/.venv-vllm` and calls `$SERVE_VENV/bin/vllm`), which avoids that failure mode; worth one clean launch-test of the public script before release, but the model+args are proven (this judge served all of Tier 3).

### Tier 4 — reproduction spot-checks (confidence, optional)
- [x] 4.1 ~200-step GRPO from released SFT init — reward climbs, no collapse
- [x] 4.2 250-record eval slice with released GRPO ckpt — per-setting coverage within noise of RESULTS
- [x] 4.3 `analyze_xpia.py` on a rebuilt-corpus eval — tables render (numbers differ from RESULTS, expected)

**Tier 4 outcome (2026-09-07):** all pass; no code fixes. **4.1** — 200-step on-the-fly GRPO from the released SFT checkpoint (`prism-qwen3.5-9b-sft.pt`) against the live judge: reward climbed +0.107 (early steps 0.646 → late 0.754), loss stayed near zero, grad norms bounded (1.4–5.5), dynamic-sampling healthy, zero judge errors — no collapse. **4.2** — 250-record judge-scored slice (n≈63/setting) with the released GRPO checkpoint reproduces RESULTS.md coverage: BN 0.984 vs 0.977, BC 0.761 vs 0.761 (exact), HO 0.609 vs 0.646, AP 0.664 vs 0.685; avg 0.7535 vs 0.767 (−0.014); hallucination avg 0.017 vs 0.020. BN/BC ~exact, HO/AP a couple points low as expected for n≈63 subsampling. **4.3** — `analyze_xpia.py` on the Tier-3.4 rebuilt-corpus rows renders every section end to end, both structurally and with the judge behaviour + provenance passes (8/8 each): per-source coverage, adversarial, follow-gated, behaviour profile, provenance split, and all breakdowns (difficulty / attacker-goal / injection-position / ib-matrix); all three sources (bipia, llmail, injecagent) present. Test jobs torn down and the judge server stopped afterward.

### Final gates (before public)
- [ ] flip `prism` GitHub public + dataset repo public; re-run 1.1–1.4 fully anonymous
- [ ] six HF repos public with matching SHAs; arXiv link live
- [ ] delete `TODO.md` from prism + squash so the internal checklist is out of public history
- [ ] Codex model/dataset cards in place

---

## External or later work

These items were discussed in the transcript but are not automatically part of the two-repository cleanup.

- [ ] Gilad's proposed PRISM paper/project website.
- [ ] Recorded slides and other media.
- [ ] Reuse approved repository visuals on the website.
- [ ] Conference demo preparation and presentation assets.
- [ ] Ignore conversational dates such as “next week” or “before October” unless Rahul establishes a current milestone.

---

## Human questions and blockers

- [?] Where is the Modal/UI code discussed in the transcript?
- [x] Where are the exact filtered training JSONLs and validity mask used for the released checkpoints? *(NFS: precomputed_data/qwen3.5-9b-last128-v5-prompt-only/relabeled_jsonl/ + valid_record_ids.json; 203,589 of 277,496 records pass the mask; same record set for all three target models.)*
- [?] Which Hugging Face organization and repository should host the training dataset?
- [?] Who has the original Zenity/Microsoft XPIA artifact?
- [?] Who will request or approve XPIA redistribution permission?
- [?] Does the current paper still report calibration kappa `0.817`, or has Table 4 been updated?
- [?] Should internal code identifiers containing “oracle” be renamed, or only public documentation and CLI text?
- [?] Where is the editable source for the paper architecture figure?
- [x] Should the first local demo be a browser UI, a CLI, or both? *(Browser UI shipped in prism/demo/; CLI still open if wanted.)*

---

## Definition of done

The public release is complete only when a new reader can:

- [ ] understand what PRISM does without first reading the paper;
- [ ] download and try the final released interpreter directly;
- [ ] find and understand the exact released training dataset;
- [ ] distinguish the target model, PRISM checkpoint, and evaluation judge;
- [ ] reproduce the final paper result without encountering historical checkpoints or step-count narratives;
- [ ] trace every distributed dataset to a defensible source, transformation, and license;
- [ ] find one canonical scoring rubric and one canonical artifact for every reported calibration result;
- [ ] understand the difference between precomputed and on-the-fly activation extraction without being told that precomputation is mandatory;
- [ ] browse both repositories without encountering internal experiment traces, abandoned prompt variants, unexplained scripts, or generated-sounding filler;
- [ ] run all documented quickstarts using commands and links that have been checked against the final repository state.

