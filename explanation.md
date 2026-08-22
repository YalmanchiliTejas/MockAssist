# Implementation Explanation

This file records why every project file and every development tool used for the
initial implementation exists. It is intentionally explicit so future changes can
preserve the project's authorization boundary.

## Legal and product decision

The requested fields are supported, but a LeetCode mass scraper is not included.
On 2026-08-15, LeetCode's official Terms of Service stated that crawling, scraping,
or spidering any part of the service is prohibited and that questions and solutions
are protected content. The package therefore operates only on local, authorized
records and requires provenance on each one. This preserves the useful archive
format without automating prohibited extraction or bypassing premium access.

## File-by-file rationale

- `.gitignore` excludes generated archives, Python caches, local environments, and
  build products so source control contains intentional source files only.
- `pyproject.toml` defines the Python version, install metadata, package discovery,
  the packaging, interactive simulator, scenario-pipeline, and interviewer-training
  console commands, and packaged prompt resources in the standard Python project
  format.
- `README.md` is the operator guide: it documents the permission boundary, input
  contract, commands, and generated archive layout.
- `src/problem_packager/__init__.py` exposes the small supported Python API and the
  package version from one location.
- `src/problem_packager/__main__.py` makes the CLI usable without installation via
  `python -m problem_packager`.
- `src/problem_packager/cli.py` parses `validate`, `package`, and `import-neetcode`
  commands, provides human-readable errors, and keeps command behavior separate
  from library code.
- `src/problem_packager/models.py` defines immutable records and centralizes strict
  validation for descriptions, constraints, hints, solutions, complexities, URLs,
  metadata, and rights provenance.
- `src/problem_packager/reader.py` reads UTF-8 JSONL, aggregates line-level errors,
  and rejects duplicate identities before anything is written.
- `src/problem_packager/render.py` converts validated records into readable Markdown
  with every requested section and each solution's complexity analysis.
- `src/problem_packager/neetcode.py` imports a local MIT-licensed NeetCode checkout.
  It verifies the license, records a commit revision, parses authored articles and
  hints, uses exact/reviewed slug mappings, emits runtime statement placeholders,
  and atomically writes JSONL. Acquisition stays separate and no network request is
  made by the importer.
- `src/problem_packager/leetcode_dataset.py` imports the selected statement-complete
  research dataset. It extracts constraints from description HTML, normalizes hints,
  splits differently formatted editorial Markdown into approaches, requires time
  and space complexity for every retained approach, and preserves layered source
  attribution and optional reference implementations.
- `src/problem_packager/normalize.py` defines the exact mock-interviewer record
  contract. It converts statement HTML to readable text, separates examples and
  constraints, creates stable snake-case approach names, derives public test pairs,
  carries reference code when available, validates exact keys and types, and writes
  output atomically. It deliberately does not synthesize missing starter or hidden
  test data.
- `src/problem_packager/exporter.py` builds the directory archive in a temporary
  sibling directory, writes normalized JSON, calculates SHA-256 checksums, and only
  replaces an existing output after an explicit `--force` flag. It refuses root,
  regular-file, and symbolic-link destinations to contain destructive scope.
- `src/candidate_simulator/__init__.py` defines the small supported simulator API so
  callers do not need to depend on internal module layout.
- `src/candidate_simulator/__main__.py` enables the demo through
  `python -m candidate_simulator` without requiring an installed console script.
- `src/candidate_simulator/models.py` defines the requested typed problem, profile,
  state, action, scenario, turn, and trajectory contracts. String enums make JSONL
  records readable while frozen dataclasses prevent actors from mutating hidden
  state accidentally.
- `src/candidate_simulator/corpus.py` streams normalized JSONL and optionally joins
  titles from the existing raw training corpus. Keeping this join separate avoids
  changing the normalized eleven-field training contract merely for CLI display.
- `src/candidate_simulator/failure_maps.py` provides deterministic editorial-
  approach-derived baseline authoring and strict cross-field validation. It checks
  identifiers, unlock and progress metadata, progression ordering, aliases,
  misconception correction targets, and bug fixtures before a map reaches runtime.
- `src/candidate_simulator/pipeline.py` selects a category-balanced 25–50-problem
  pilot, preserves the five hand-reviewed maps as overrides, fingerprints authoring
  inputs, records generator provenance, audits scenario counts, and atomically
  persists catalogs. This makes scaling reproducible and keeps heuristic records
  distinguishable from reviewed ones.
- `src/candidate_simulator/pipeline_cli.py` exposes the batch authoring and audit
  pipeline as `scenario-pipeline` without coupling it to the interactive demo.
- `src/candidate_simulator/profiles.py` keeps the eight candidate archetypes in one
  reviewed catalog. Numeric traits are bounded from zero to one and are consumed by
  deterministic transition policy, not disclosed in public observations.
- `src/candidate_simulator/scenarios.py` contains reviewed failure maps for the five
  MVP problems, seeded profile-to-state construction, normalized-corpus loading,
  and an optional OpenAI-compatible editorial-to-failure-map authoring helper. The
  helper runs before an episode; runtime actors never receive its editorial input.
- `src/candidate_simulator/transitions.py` is the authority for concept unlocking,
  misconception resolution, planned-bug evidence, phase changes, allowed actions,
  and progress. Keeping those decisions outside generated text is the central
  anti-omniscience guarantee. Code and example-test progress additionally require a
  valid known approach so an interviewer cannot manufacture progress by demanding
  code before the candidate understands a solution.
- `src/candidate_simulator/validators.py` parses structured candidate JSON and
  rejects disallowed phase actions, unknown-concept use, premature bug fixes,
  malformed code/complexity actions, and disclosure of private simulator fields.
- `src/candidate_simulator/environment.py` coordinates one seeded episode. It
  projects an allow-listed actor context, retries invalid generations twice, uses a
  deterministic fallback, records state before and after every turn, and returns a
  separate public observation with no hidden knowledge or evaluator state.
- `src/candidate_simulator/storage.py` atomically writes one complete trajectory per
  JSONL line and reconstructs the typed nested objects when loading. Explicit
  replacement and symlink checks protect existing artifacts.
- `src/candidate_simulator/cli.py` supplies the requested runnable demo. It supports
  interactive problem/profile selection, plain-message `ASK` actions, full JSON
  interviewer actions, scripted or compatible-LLM actors, seeds, and trajectory
  export without adding a frontend.
- `src/candidate_simulator/actors/__init__.py` exposes the supported actor types from
  one import location.
- `src/candidate_simulator/actors/base.py` defines the pluggable actor interface and
  the exact public context allow-list. Its deliberately narrow model is auditable
  and prevents accidental editorial or unknown-concept injection.
- `src/candidate_simulator/actors/scripted.py` provides reproducible natural speech,
  small code templates, complexity claims, and the safe validation fallback. It
  makes tests and demos work without credentials or network access. Explicit
  `REQUEST_TEST` and `REQUEST_COMPLEXITY` actions are handled before generic
  preferences so the deterministic actor responds to the interviewer's requested
  activity when multiple candidate actions are valid in the same phase.
- `src/candidate_simulator/actors/mock.py` returns queued raw strings and records
  contexts so malformed JSON, retries, and privacy boundaries can be tested without
  calling a model.
- `src/candidate_simulator/actors/openai_compatible.py` implements the optional
  `/chat/completions` adapter with `urllib`, structured-output mode, authorization,
  timeout handling, and no third-party runtime dependency.
- `src/candidate_simulator/training.py` implements the first interviewer baseline:
  fixed public action choices, deterministic action grounding, a public-only state
  projection, evaluator-only reward, tabular Q-learning, seeded evaluation, and
  atomic policy persistence. A small Q-table was chosen before an RL framework so
  environment leakage and reward shortcuts remain directly inspectable.
- `src/candidate_simulator/training_cli.py` exposes repeatable training parameters
  and prints untrained versus trained evaluation metrics so a run cannot be judged
  only by its training reward.
- `src/candidate_simulator/prompts/__init__.py` makes the prompt directory a Python
  resource package.
- `src/candidate_simulator/prompts/candidate_actor.txt` keeps the actor behavior and
  exact JSON-output requirements inspectable outside Python code. It explicitly
  requires state fidelity, misconception and bug persistence, natural interview
  behavior, and nondisclosure of private state.
- `examples/problems.jsonl` is an original, CC0 demonstration record showing the
  complete input contract without copying a third-party coding challenge.
- `examples/neetcode-aliases.json` records a small set of manually reviewed
  LeetCode-to-NeetCode filename differences. Keeping aliases explicit prevents an
  approximate matcher from silently attaching the wrong solution.
- `tests/test_reader.py` covers successful reading, aggregated invalid input,
  provenance enforcement, and duplicate detection.
- `tests/test_exporter.py` covers requested Markdown sections, checksum correctness,
  replacement refusal, explicit forced replacement, code-fence safety, and
  destructive-path guards.
- `tests/test_neetcode.py` uses an original miniature checkout fixture to cover
  article, hint, complexity, alias, license, provenance, and atomic-output behavior
  without embedding third-party test content.
- `tests/test_leetcode_dataset.py` uses short original fixtures to cover constraint
  extraction, editorial-format variations, complexity cleanup, completeness
  filtering, hints, provenance, and reference-code preservation.
- `tests/test_normalize.py` uses an original HTML fixture to cover prompt/example
  separation, structured examples, the exact eleven-field contract, approach-name
  normalization, public tests, reference code, and safe output replacement.
- `tests/test_candidate_simulator.py` verifies every requested state invariant,
  actor retry/fallback behavior, deterministic seeds, public/private separation,
  trajectory round trips, progress tracking, explicit request dispatch, and all 40
  curated problem/profile combinations using the existing local normalized corpus.
- `tests/test_scenario_pipeline.py` verifies title joins, representative selection,
  golden-map preservation, invalid-map rejection, JSONL round trips, and all 200
  combinations in the minimum 25-problem/eight-profile pilot.
- `tests/test_interviewer_training.py` verifies that policy features exclude private
  state, action grounding does not depend on candidate hidden state, premature code
  requests cannot create progress, identical seeds reproduce training, and saved
  policies reload exactly.
- `data/candidate-simulator/pilot-failure-maps.jsonl` is the generated 32-problem
  authoring artifact. Storing maps separately from normalized problem data keeps
  editorials out of runtime actor context and enables reviewed replacements without
  rewriting the source corpus.
- `data/candidate-simulator/pilot-validation.json` is the machine-readable count,
  category, generator, and error audit used to gate the pilot artifact.
- `artifacts/interviewer-policy.json` is the reproducible seed-42 baseline Q-table
  plus its public feature contract, configuration, and evaluation report.
- `artifacts/README.md` records the policy's scope, checksum, metrics, and warning
  that simulator performance is not equivalent to real interview quality.
- `data/normalized-training.jsonl` is the final training-facing representation. It
  contains one exact-schema JSON object per line for all 701 complete primary
  records.
- `data/leetcode-training.jsonl` is now the primary training corpus because every
  retained record has a statement, parsed constraints, and at least one solution
  approach with both complexity values.
- `data/neetcode-training.jsonl` is the generated training-ready corpus requested by
  the user. It contains licensed NeetCode reasoning and code plus placeholders for
  statements that the runtime host must provide.
- `data/README.md` records corpus counts, exact source revision, reproduction steps,
  attribution, checksums, source limitations, and which corpus is primary.
- `data/LICENSE.neetcode` preserves the full upstream MIT notice as required when
  redistributing copies or substantial portions of the imported material.
- `changelog.md` provides chronological, user-visible release and compliance notes.
- `explanation.md` provides the requested audit trail for every file and tool choice.

## Tool-by-tool rationale

- Repository inspection used `pwd`, `rg --files`, `ls`, and `find` through the shell
  execution tool. These established that the workspace was empty and checked for
  repository-specific `AGENTS.md` instructions before creating files. `rg` was used
  first because it is the preferred fast file-search tool; `find` checked parent
  directories after no repository files were returned.
- The asynchronous wait tool collected the completion of the parent-directory
  instruction search after its first shell call yielded.
- The web search tool consulted LeetCode's official Terms of Service because access
  rules are external, legally relevant, and can change. The terms page was treated
  as the primary source for the compliance decision.
- The planning tool tracked endpoint/terms verification, implementation, tests, and
  documentation so validation was not omitted from this multi-file build.
- `mkdir` created the Python source, test, and example directories before patching;
  directory creation is a narrow, reversible workspace operation.
- The patching tool created source and documentation files with reviewable unified
  diffs, which avoids opaque or accidental file rewrites.
- Python's built-in `unittest` runner exercises the validation and archive behavior
  without introducing a test dependency. The CLI is also run against the original
  example to verify the installed-module entry path and inspect an actual archive.
- `compileall` performs a separate syntax/import compilation check across source and
  tests. Final `rg`/`find` file listings and `wc` byte counts provide a read-only
  scope audit of source files and the generated sample archive.
- For the candidate-simulator MVP, `rg`, `sed`, and short read-only Python commands
  inspected the existing package conventions and selected five records actually
  present in the normalized corpus. `mkdir` created only the new package resource
  directories. The planning tool tracked the model, state machine, actor boundary,
  environment, tests, and documentation as distinct completion gates. The patching
  tool added every implementation and documentation change as a reviewable diff.
  `compileall`, the module `--list` command, a seeded multi-turn audit, and the
  built-in `unittest` runner verify imports, packaged prompt discovery, all 40
  scenario combinations, persistence, and behavioral invariants without installing
  dependencies or contacting an LLM endpoint.
- For the scenario/training expansion, `rg` and `sed` inspected existing contracts,
  CLI conventions, tests, and documentation before edits. Short read-only Python
  audits measured corpus approach/hint/title coverage and reviewed generated pilot
  IDs, categories, metrics, and policy state counts. The patching tool added the
  authoring, validation, training, tests, packaging, and documentation changes as
  reviewable diffs.
- `mktemp` provided a unique pilot dry-run directory under `/private/tmp`; the
  pipeline wrote there first so structural and category problems could be corrected
  before creating permanent data artifacts. The scenario and training CLIs then
  generated their deterministic workspace artifacts through atomic writers.
- The built-in `unittest` runner verifies the new 25-problem minimum pilot and
  training invariants without an external ML dependency. `compileall` checks syntax
  and imports. A second 500-episode seed-42 run wrote only to `/private/tmp`, and
  `shasum` confirmed that its policy bytes matched the checked-in artifact exactly.
  `rg` separately confirmed that forbidden hidden-state feature names do not occur
  in the saved policy. `wc` and JSON read-only summaries audited artifact sizes,
  counts, generator provenance, category balance, and reported metrics.
- Explicit `rm -rf` cleanup removed only reviewed Python `__pycache__` directories
  created during verification under the two source packages and tests. Their
  absolute paths were enumerated; they are generated, ignored artifacts and are not
  part of the deliverable.

No dependency installer, browser automation, account cookie, LeetCode API, or live
scraping tool is used. The runtime uses only Python's standard library to minimize
supply-chain and deployment overhead.

## Source-repository audit (2026-08-15)

The follow-up investigation considered whether open repositories could replace a
direct LeetCode scraper:

- `neetcode-gh/leetcode` carries an MIT license and is a strong source for content
  authored by its contributors: solution code, solution articles, hints, and time
  and space complexity. Its metadata links to LeetCode rather than copying the
  statement, and its articles generally have no Description or Constraints section.
- `doocs/leetcode` carries CC-BY-SA-4.0 and has a convenient complete Markdown
  structure. Its contributor-authored explanations can be reused with attribution
  and ShareAlike compliance. The included problem statements appear to reproduce
  LeetCode text, however, and an open-source license cannot grant rights the project
  does not own.
- `walkccc/LeetCode` and `kamyu104/LeetCode-Solutions` carry MIT licenses and are
  useful solution/complexity sources, but do not solve the statement-rights gap.

The local NeetCode audit found 450 indexed LeetCode metadata records, 772 authored
articles (771 with explicit complexity headings), 150 hint files, and 396 Python
solution files in the selected checkout. Exact filenames connect only 346 articles
and 76 hints to indexed LeetCode slugs. An importer should therefore use exact
matches plus a reviewed alias map; fuzzy matching would risk silently packaging the
wrong explanation.

### Additional tools used for this audit

- The GitHub orientation skill established a connector-first repository-audit
  workflow and emphasized resolving concrete repository scope before acting.
- Official GitHub pages and raw license files were opened through the web tool to
  verify current repository structures and license text from primary sources.
- A shallow, blob-filtered, sparse `git clone` placed the NeetCode repository in
  `/private/tmp` for read-only format inspection without adding third-party material
  to this project. `git sparse-checkout` fetched only articles, hints, Python files,
  and top-level metadata; its sandboxed retry failed on DNS, so the approved network
  retry completed the same narrowly scoped operation.
- `sed` inspected representative source and instruction files. `rg` located
  headings, links, and filename relationships. `find` and `wc` measured source
  coverage. A short read-only Python expression compared JSON metadata slugs with
  repository filenames because set intersection is clearer and less error-prone
  than a multi-stage shell pipeline for that structured data.
- The asynchronous wait tool collected completion of the approved sparse checkout
  after the command yielded.
- The new `import-neetcode` CLI was run first into a unique temporary directory,
  validated with the existing CLI, and summarized with a read-only Python JSON
  expression and byte count. After those checks passed, the same deterministic
  command generated the workspace corpus. `git rev-parse` independently confirmed
  the recorded commit, and `mkdir` created only the intended `data` directory.
- Final end-to-end verification packaged all 352 records into a temporary archive.
  `shasum` recorded the training file's SHA-256 digest, while `find`, `wc`, and a
  short JSON read confirmed 352 problem directories and 706 checksummed archive
  files. The locale warning emitted by `shasum` did not affect its successful digest.
- For the replacement-source audit, the GitHub orientation skill framed the
  repository comparison, while web search and primary repository/dataset pages
  verified live schemas, intended use, licenses, and source attribution. `curl`
  downloaded the 20.4 MB raw JSON only to `/private/tmp`; the sandboxed attempt hit
  DNS restrictions and the user-approved retry succeeded. Read-only Python schema
  and formatting audits measured field coverage and revealed multiple editorial
  Markdown variants before the importer was finalized. The importer was tested in a
  unique temporary directory before generating the primary workspace corpus, and
  `shasum` recorded its reproducible SHA-256 digest.
- The normalization CLI was first run into a `mktemp`-created directory. A read-only
  Python audit checked all 701 key sequences and counted statements, editorials,
  examples, approaches, hints, starter code, reference solutions, and hidden tests.
  Only after this dry run and 24 passing tests did the same command generate the
  workspace corpus. `shasum` then recorded the normalized file's SHA-256 digest; its
  locale warning did not affect the successful result.
