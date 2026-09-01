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

The Modal training path uses the local `Qwen/Qwen3.5-4B` model for candidate
responses. Set `MOCKASSIST_CANDIDATE_MODEL` to change the Hugging Face model ID.
The model is loaded lazily in each worker and receives only the allow-listed public
actor context; it never receives the editorial, unknown concepts, unlock thresholds,
planned bug, or evaluator state.

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

### GRPO checkpoint evaluation

Reserve a problem-level test partition when training the GRPO interviewer. This
keeps all candidate profiles for a held-out problem out of training:

```bash
modal run src/modal_train.py::train --run-name run-001 --heldout-fraction 0.1
modal run --detach src/modal_train.py::evaluate \
  --run-name run-001 \
  --heldout-fraction 0.1 \
  --evaluation-name evaluation-smoke \
  --profiles strong \
  --seeds 0
```

The Modal evaluator defaults to this single-problem, single-profile, single-seed
smoke test on one `L4`. The 9B interviewer stays on the GPU while a 0.8B candidate
simulator runs on CPU; both generations are capped at 256 tokens. The function has
an explicit one-hour timeout and resource allocation whose published-rate maximum
is approximately $1.15. A preflight rejects budgets above $30 or below that maximum;
the default budget is $5. It resumes its JSONL by default, so restarting the
same evaluation name skips completed scenarios. Use this detached evaluation as a
format and adapter smoke test. Check
that malformed-action rate is near zero and that the adapter loads without missing
keys before explicitly increasing `--max-problems`, profiles, and seeds for a full
held-out evaluation.

Both commands must use the same `--heldout-fraction` and `--split-salt` (the
default is `mockassist-v1`). Evaluation produces
`evaluation/evaluation-rollouts.jsonl` and `evaluation/evaluation-metrics.json`
plus TensorBoard events under `evaluation/tensorboard` inside the checkpoint
volume. It evaluates deterministic rollouts over every
held-out problem, the `strong`, `average`, and `nervous` candidate profiles, and
three fixed seeds. Metrics include reward, solution-reached rate, timeout rate,
elapsed time, hints, malformed action rate, format-retry rate, premature-end rate,
interviewer-end rate, and 95% bootstrap confidence intervals, with profile and
category breakdowns. Rollouts are flushed after every episode, so a stopped run
retains partial results and its logs report explicit episode progress.
Pass `--baseline-checkpoint <base-model-or-adapter>` to `evaluate` to run the
same scenarios against a baseline and include trained-minus-baseline deltas.

The environment keeps the `+0.05` valid-turn reward. Candidate Python code runs in
a short-lived, resource-limited local child process by default and receives no
direct pass/submission reward. The child first detects syntax errors, top-level
exceptions, timeouts, and crashes, then invokes ordinary `Solution` methods or
stateful design-class operation sequences. Outputs are checked against the dataset's
Python reference implementation on published and deterministically derived private
inputs. Design problems without a Python reference use their published operation
outputs plus safe derived replay cases. If no executable oracle can be constructed,
the result is an infrastructure error and never a pass. A failed run is returned to
the interviewer; relaying
that failure and requesting corrected code earns a one-time `+0.03` recovery reward
for that failure. Successful termination
earns `+2.0` only after code has passed the sandbox and a complexity claim has been
observed. Premature termination earns `-1.0`, timeout without a solution earns
`-0.5`, and an
unrepairable malformed evaluation action earns `-1.0` without entering the normal
terminal path. Training completions that stop without ending the environment also
earn `-1.0`.

No Fly app or droplet configuration is needed for the POC smoke evaluation. The
local executor uses Python isolated mode, a sanitized environment, a temporary
working directory, and time/memory/process/output limits. It is not a hardened
multi-tenant sandbox and should be replaced by container isolation before exposing
the evaluator to untrusted users.

Remote execution remains available as an explicit option:

```bash
export MOCKASSIST_CODE_EXECUTOR="remote"
export MOCKASSIST_CODE_EXECUTOR_URL="https://<runner>/sandbox/execute"
export MOCKASSIST_CODE_EXECUTOR_TOKEN="<runner-token>"  # when required
```

The existing Gauntlet names are also supported: when `SANDBOX_RUNNER_URL` is set,
MockAssist calls `${SANDBOX_RUNNER_URL}/sandbox/execute` and reads the optional
`SANDBOX_RUNNER_TOKEN`. The runner accepts Python source, a timeout, and public
problem context and returns `status`, `stdout`, `stderr`, `exit_code`, and
`timed_out`. Runner/network errors are recorded as infrastructure failures and do
not qualify for the recovery reward. Candidate source and normalized execution
results are retained in both training and evaluation rollouts.
For Modal jobs, pass `--code-executor-url` and, when needed,
`--code-executor-token`; the function places them in the worker environment and
does not add them to the spawned training/evaluation command line.

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
