# Problem Packager

Problem Packager turns authorized JSONL problem records into a deterministic,
browsable archive. Each generated problem page contains its description, hints,
constraints, solutions, and the time and space complexity of every solution.

This repository intentionally does **not** contain a LeetCode scraper. LeetCode's
current terms prohibit crawling, scraping, and spidering its service and identify
questions and solutions as protected content. Use this tool only with content you
created, licensed, received through an authorized export/API, or otherwise have
permission to process.

## Quick start

Python 3.11 or newer is required. The application itself has no third-party
runtime dependencies.

```bash
python -m pip install -e .
problem-packager validate examples/problems.jsonl
problem-packager package examples/problems.jsonl --output dist/problems
```

Without installing it, run the same commands with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m problem_packager validate examples/problems.jsonl
PYTHONPATH=src python -m problem_packager package examples/problems.jsonl \
  --output dist/problems
```

Use `--force` only when you explicitly want to replace an existing output
directory.

## Training corpora

The primary training input is
[`data/leetcode-training.jsonl`](data/leetcode-training.jsonl). It contains 701
complete records with embedded statements, parsed constraints, hints, and 1,540
solution approaches with per-approach time and space complexity. It was built from
the `Alishohadaee/leetcode-problems-dataset` research dataset.

[`data/neetcode-training.jsonl`](data/neetcode-training.jsonl) remains available as
an optional supplemental reasoning corpus, but it is not a default training input
because it does not contain statements or authoritative constraints.

After downloading the upstream `raw_data/leetcode_problems.json`, rebuild the
primary corpus with:

```bash
PYTHONPATH=src python -m problem_packager import-leetcode-dataset \
  /path/to/leetcode_problems.json \
  --revision <dataset-revision> \
  --output data/leetcode-training.jsonl \
  --force
```

The importer reads local JSON and performs no network requests. See
[`data/README.md`](data/README.md) for field coverage, checksums, attribution, and
source limitations.

## Normalize for mock-interviewer training

[`data/normalized-training.jsonl`](data/normalized-training.jsonl) is the final
normalized corpus. Each line is one object with exactly these fields:

```json
{
  "problem_id": "...",
  "statement": "...",
  "constraints": [],
  "examples": [],
  "starter_code": "",
  "editorial": "...",
  "approaches": [
    {
      "name": "brute_force",
      "explanation": "...",
      "time_complexity": "O(n^2)",
      "space_complexity": "O(1)"
    }
  ],
  "hints": [],
  "reference_solution": "",
  "public_tests": [],
  "hidden_tests": []
}
```

Rebuild it with:

```bash
PYTHONPATH=src python -m problem_packager normalize \
  data/leetcode-training.jsonl \
  --output data/normalized-training.jsonl \
  --force
```

Examples are extracted as `{input, output, explanation?}` objects. Public tests are
derived as `{input, expected_output}` pairs. The current upstream source has no
starter-code or hidden-test fields, so those values remain empty rather than being
invented.

## Candidate simulator MVP

The `candidate_simulator` package runs state-controlled mock candidates. The
interactive demo has five hand-reviewed problems: Two Sum, Merge Intervals, Move
Zeroes, Top K Frequent Elements, and Binary Search. Each problem supports eight
profiles, from a strong independent candidate to a strong debugger. Scenario
state—not the actor—decides which concepts are known, when hints unlock concepts,
when misconceptions clear, and when planned bugs are detected.

List the available demo choices:

```bash
PYTHONPATH=src python -m candidate_simulator --list
```

Run an interactive, reproducible scripted session:

```bash
PYTHONPATH=src python -m candidate_simulator \
  --problem 1 \
  --profile intermediate_bruteforce \
  --seed 42 \
  --output trajectory.jsonl
```

A plain input line becomes an `ASK`. Send JSON for a structured action:

```json
{"action_type":"HINT","message":"Could lookup be faster?","target_concept":"hash_map_complements","hint_level":2}
```

Use `--actor openai --base-url <compatible-v1-url> --model <model>` for an
OpenAI-compatible chat-completions endpoint. `OPENAI_BASE_URL`, `OPENAI_MODEL`, and
`OPENAI_API_KEY` are also recognized. The endpoint receives only the allow-listed
public actor context; it never receives the editorial, unknown concepts, unlock
thresholds, planned bug, or evaluator state.

The final JSONL trajectory contains the private before/after states for offline
analysis. Those fields are deliberately excluded from every interviewer observation.

## Scalable scenario pilot

Build a deterministic, representative pilot from 25–50 normalized problems:

```bash
PYTHONPATH=src python -m candidate_simulator.pipeline_cli \
  --count 32 \
  --output data/candidate-simulator/pilot-failure-maps.jsonl \
  --report data/candidate-simulator/pilot-validation.json \
  --force
```

The checked-in pilot contains 32 problems across 16 broad algorithm categories and
supports all eight profiles, for 256 scenarios. The five original maps remain
hand-reviewed overrides. The other 27 are deterministic heuristic baselines derived
from normalized editorial approaches. Every record includes its generator label and
a SHA-256 source fingerprint, and every map is checked for identifiers, complete
unlock/progress metadata, monotonic progress, valid misconception targets, aliases,
and bug consistency.

Heuristic validation establishes structural correctness, not expert correctness.
Review these maps before treating them as production training labels. The generator
uses a deliberately incomplete universal bug fixture where it cannot safely derive a
problem-specific faulty implementation.

## First interviewer training loop

Train the dependency-free tabular Q-learning baseline:

```bash
PYTHONPATH=src python -m candidate_simulator.training_cli \
  --catalog data/candidate-simulator/pilot-failure-maps.jsonl \
  --episodes 500 \
  --evaluation-episodes 100 \
  --seed 42 \
  --output artifacts/interviewer-policy.json \
  --force
```

The policy sees only phase, whether public code exists, public example-failure count,
hints used, remaining-time bucket, and the last public action from each participant.
It never receives candidate knowledge, unknown concepts, misconceptions, planned
bugs, unlock thresholds, or progress. A separate evaluator may use hidden
before/after states to calculate reward, just as a training environment can expose a
reward without exposing its private answer key to the policy.

The v0 reward favors candidate progress, misconception resolution, bug discovery,
testing, and complexity discussion. It penalizes stronger hints, repeated responses,
validation failures, turns, and incomplete termination. The checked-in seed-42 run
learned 381 public states. On its deterministic 100-episode evaluation it improved
from a 2.44/7 untrained average to 6.66/7, with 85% reaching level 7. These are
simulator metrics, not evidence of real interview quality.

This stage still does not execute untrusted candidate code, generate hidden tests,
fine-tune an LLM, or establish that the 27 heuristic maps are semantically expert
quality. The scripted actor also remains less natural than the compatible-LLM actor.

## Starter practice list

The repo includes a public NeetCode 150 starter backlog at
[`examples/neetcode_150.md`](examples/neetcode_150.md). This is the first study
track to work through when using the project as a local practice archive.

## Input format

The input is UTF-8 JSON Lines: one JSON object per line. Blank lines are ignored.
Every record must include:

- `id`, `slug`, `title`, `difficulty`, `source_url`, and `description`
- a non-empty `constraints` array and a `hints` array
- a non-empty `solutions` array; every solution needs `title`, `explanation`,
  `time_complexity`, and `space_complexity`
- a `rights` object containing `source_name`, `license`, `attribution`, and a
  concrete `permission_basis`

Optional solution fields are `language` and `code`. Optional record metadata goes
under `metadata` and must be a JSON object. See
[`examples/problems.jsonl`](examples/problems.jsonl) for a complete record.

## Output format

`package` creates:

```text
output/
├── README.md
├── index.json
├── manifest.json
└── problems/
    └── <id>-<slug>/
        ├── README.md
        └── metadata.json
```

`manifest.json` includes a SHA-256 checksum for every other generated file so an
archive can be checked for accidental changes. Output is deterministic for the
same input bytes and packager version.

## Adding an authorized source

Convert the authorized source to the documented JSONL contract, then validate and
package it. Keeping acquisition outside this project makes the permission boundary
visible: the `rights.permission_basis` field travels with every problem, and this
tool never sends network requests or accepts browser cookies.

See [`explanation.md`](explanation.md) for the reasoning behind every project file
and development tool, and [`changelog.md`](changelog.md) for release history.
