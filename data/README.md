# Training Corpora

## Candidate-simulator pilot

`candidate-simulator/pilot-failure-maps.jsonl` is the checked-in 32-problem
failure-map pilot. It combines five hand-reviewed maps with 27 deterministic
editorial/approach-derived heuristic maps. The selection spans 16 broad categories
and expands to 256 problem/profile scenarios across the eight candidate profiles.
Each record stores a generator label and source fingerprint so reviewed overrides
and regenerated records remain distinguishable.

`candidate-simulator/pilot-validation.json` records the structural audit: 32 valid
problems, eight profiles per problem, 256 total scenarios, generator counts,
category counts, and no validation issues. Structural validity does not replace
expert semantic review of the heuristic maps.

SHA-256:

- `pilot-failure-maps.jsonl`:
  `d412e0c88f38efb7667af550ff94770ad1ff0566610e9032eb2ae0e1eec0f3e2`
- `pilot-validation.json`:
  `8cdddac7d584d3ebeaa4cc5b511bff27b5f58f06fc7a57c358922216b0a4e844`

## Final normalized corpus: `normalized-training.jsonl`

This is the intended mock-interviewer training input. It contains 701 JSONL records
with the exact normalized field contract documented in the project README.

Coverage after normalization:

- 701 records with exact schema conformance
- 701 non-empty statements and editorials
- 1,540 structured approaches with individual time and space complexity
- 1,516 examples from 627 problems, also exposed as public test pairs
- 1,042 hints across 468 problems
- 226 reference Python solutions
- 0 starter-code values because the selected source does not provide that field
- 0 hidden tests because no hidden-test source was available

SHA-256:
`6d21858c7271ef047516539fcf381b2d212e4418f101dc65304c4b3c12991e72`.

## Primary: `leetcode-training.jsonl`

This is the default structured corpus for training or evaluating the mock coding
interviewer. It was generated from the
[`Alishohadaee/leetcode-problems-dataset`](https://huggingface.co/datasets/Alishohadaee/leetcode-problems-dataset)
raw JSON revision `e429e07`.

Coverage:

- 701 complete packaged records from 3,549 source rows
- 701 embedded descriptions with separately parsed constraints
- 1,540 solution approaches, each with time and space complexity
- 1,042 hints across 468 records
- 226 records with supplemental Python reference code
- 719 source rows skipped without descriptions
- 107 otherwise eligible rows skipped without parseable constraints
- 2,022 otherwise eligible rows skipped without parseable per-approach complexity

SHA-256:
`268b08aa37aebf1c05752b2a63df3ff79e0d5f986c29bebdd7cbe359a4f2a88a`.

The dataset card labels its repository MIT and states that underlying problem and
solution content belongs to LeetCode, with additional material from LeetCodeHelp.
It designates the dataset for educational and research purposes. Each output record
preserves that layered attribution and the upstream problem URL. Review applicable
source terms before redistribution or commercial use.

## Supplemental: `neetcode-training.jsonl`

This earlier corpus contains 352 MIT-attributed NeetCode reasoning records, 1,267
approaches, and 283 hints. It is retained for optional experiments but excluded from
the default training input because its descriptions and constraints are runtime
placeholders. Its upstream license is preserved in
[`LICENSE.neetcode`](LICENSE.neetcode).

## Reproduction

After downloading `raw_data/leetcode_problems.json` from the primary dataset:

```bash
PYTHONPATH=src python -m problem_packager import-leetcode-dataset \
  /path/to/leetcode_problems.json \
  --revision e429e07 \
  --output data/leetcode-training.jsonl \
  --force

PYTHONPATH=src python -m problem_packager validate \
  data/leetcode-training.jsonl

PYTHONPATH=src python -m problem_packager normalize \
  data/leetcode-training.jsonl \
  --output data/normalized-training.jsonl \
  --force
```
