# Changelog

All notable changes to this project are recorded here.

## Unreleased

### Fixed

- Aligned checkpoint evaluation with the training-time Qwen tool-call template,
  disabled thinking during structured generation, added one format-repair attempt,
  and stopped malformed output from receiving a successful terminal reward.
- Loaded Qwen interviewer adapters through the same multimodal architecture used
  during training and made evaluation fail on missing LoRA keys instead of silently
  evaluating a partial adapter.
- Made the `+2.0` terminal reward conditional on observed code and complexity,
  penalized premature endings, timeouts, malformed evaluation actions, and
  incomplete training episodes, and skipped unnecessary candidate generation after
  an interviewer `END` action.
- Fixed the deterministic scripted actor selecting `MODIFY_CODE` when the
  interviewer explicitly sent `REQUEST_TEST`. Explicit test and complexity
  requests now take precedence over generic actor action preferences.
- Prevented code-writing and simulated-example actions from granting progress when
  the candidate has no valid, known approach. This closes a shortcut that an
  interviewer policy could exploit by rushing weak candidates directly into code.

### Added

- Added shared interviewer prompts, explicit code/complexity/end-state tracking,
  per-episode evaluation persistence and progress logs, format-retry and
  premature-end metrics, direct evaluation TensorBoard events, and
  `--max-problems` smoke evaluations.
- Added a batch scenario pipeline that joins normalized problems with title
  metadata, selects a deterministic 25–50-problem category-balanced pilot,
  preserves manual maps as overrides, derives heuristic maps from authored
  approaches, fingerprints source material, validates cross-field invariants, and
  atomically exports JSONL plus an audit report.
- Added a checked-in 32-problem pilot spanning 16 broad categories. Its five
  hand-reviewed and 27 heuristic maps expand to 256 scenarios across eight
  profiles, with explicit generator provenance and a clean structural audit.
- Added the first interviewer-training loop: a public-observation-only tabular
  Q-learning baseline, fixed high-level action grounding, evaluator-only hidden
  state reward, seeded training/evaluation, policy serialization, and CLI.
- Added a reproducible seed-42, 500-episode policy artifact. On the current scripted
  simulator's 100-episode evaluation, mean progress moved from 2.44/7 to 6.66/7 and
  level-7 completion from 0% to 85%; these are simulator rather than real-interview
  quality metrics.
- Added ten pipeline/training tests covering metadata joins, selection diversity,
  200 scenario combinations, catalog validation and persistence, public-only policy
  state, action-grounding isolation, reward-shortcut prevention, reproducible
  training, and policy persistence. The complete project now has 46 tests.
- Added a state-controlled candidate-simulator MVP with immutable typed models,
  deterministic concept/misconception/bug transitions, progress levels 0–7, public
  observation filtering, and complete private trajectory auditing.
- Added eight reusable candidate profiles and reviewed failure maps for Two Sum,
  Merge Intervals, Move Zeroes, Top K Frequent Elements, and Binary Search. All 40
  problem/profile combinations run end-to-end from the normalized corpus.
- Added a pluggable actor boundary with deterministic scripted, queue-backed mock,
  and OpenAI-compatible chat-completions implementations. Actor inputs are
  allow-listed, structured JSON is validated, and two failed regenerations trigger
  a safe scripted fallback.
- Added the `candidate-simulator` interactive CLI for selecting a demo problem and
  profile, entering plain or structured interviewer actions, using seeded sessions,
  and exporting a lossless JSONL trajectory.
- Added twelve core candidate-simulator tests covering gated concept learning,
  persistent misconceptions and bugs, profile differences, reproducibility,
  observation privacy, regeneration/fallback behavior, storage round trips, and the
  complete five-problem/eight-profile matrix.
- Added an exact mock-interviewer normalization schema and `normalize` CLI command.
  The normalizer separates statement text, constraints, examples, approach-level
  explanations and complexity, hints, reference code, and public test pairs.
- Added the final 701-record `normalized-training.jsonl` corpus with 1,540
  approaches and 1,516 extracted examples. Unavailable starter code and hidden tests
  remain empty instead of being fabricated.
- Added four normalization tests for statement separation, example extraction,
  exact schema conformance, snake-case approach names, reference solutions, public
  tests, and atomic output protection.
- Added an importer for the `Alishohadaee/leetcode-problems-dataset` raw JSON. It
  extracts constraints from statement HTML, splits official editorials into
  approaches, and retains only approaches with their own time and space complexity.
- Added a new primary 701-record, statement-complete training corpus with 1,540
  analyzed solution approaches and 1,042 hints. The earlier NeetCode corpus is now
  explicitly supplemental rather than a default training source.
- Added three dataset-import tests covering HTML constraints, varying editorial
  Markdown structure, completeness filtering, hints, provenance, and reference code.
- Added a local NeetCode repository importer that verifies the expected MIT
  license, auto-detects the source revision, extracts numbered solution approaches,
  retains a selected code language, parses time/space complexity and hints, and
  emits validated JSONL without network access.
- Added reviewed slug aliases rather than unsafe fuzzy matching.
- Added a generated 352-record NeetCode training corpus containing 1,267 solution
  approaches and 283 hints, with the complete upstream MIT notice and reproducible
  source commit.
- Added runtime-statement metadata and placeholders so a future plugin can inject
  the active prompt transiently without placing LeetCode statements in training.
- Added six importer/parser/output safety tests.

### Research

- Audited the MIT-licensed `neetcode-gh/leetcode` repository as a potential local
  import source. It provides extensive authored solutions, hints, and complexity
  analysis, but does not redistribute problem descriptions or constraints.
- Audited `doocs/leetcode`, `walkccc/LeetCode`, and
  `kamyu104/LeetCode-Solutions`. Their licenses can cover contributor-authored
  solutions, but cannot automatically grant rights to third-party LeetCode problem
  statements copied into a repository.
- Identified the need for explicit NeetCode-to-LeetCode slug aliases: silently
  fuzzy-matching differently named articles could attach a solution to the wrong
  problem.

## 0.1.0 - 2026-08-15

### Added

- Added a dependency-free Python CLI for validating authorized JSONL problem data.
- Added deterministic Markdown and JSON archive generation.
- Added required rights provenance for every input record.
- Added per-solution time and space complexity validation and rendering.
- Added checksummed manifests, duplicate detection, safe replacement behavior
  (including root, file, and symbolic-link guards), an original example record,
  and automated tests.

### Compliance

- Kept acquisition out of the implementation because LeetCode's current terms
  explicitly prohibit crawling, scraping, and spidering its service and restrict
  copying its questions and solutions.
