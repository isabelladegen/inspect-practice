# inspect-practice

Running [Inspect](https://inspect.aisi.org.uk) evals against local models served by Ollama,
and rescoring stored answers with different grading setups.

## What it does

- **Runs evals** of several local models on a small security Q&A dataset
  (Inspect's `security_guide` example).
- **Rescores stored answers** without regenerating them: answers are replayed
  from earlier logs and graded again by a model grader.
- **Grades along two axes**: *completeness* against the reference answer, and
  *additions* beyond it (with N for "no additions", excluded from means).
- **Grades in two ways**: one answer at a time, or all models' answers to the
  same question together, anonymised and in a seeded random order.
- **Summarises results** per grader and model, as CSV tables.

## Architecture

- **Eval side:** this repo. Python 3.12, managed by uv, with Inspect.
- **Model side:** Ollama, installed system-wide, started on demand and
  listening on `localhost:11434`. Inspect talks to it over HTTP.
- Models are defined once in `src/utils/models.py` and pulled with
  `src/utils/pull_models.py`. Weights live in `~/.ollama/models`, outside this repo.

## Layout

```
inspect-practice/
├── pyproject.toml / uv.lock
├── src/
│   ├── utils/          # config (the only place that knows paths), models, Inspect adapters
│   ├── tasks/          # Inspect tasks, including replaying stored answers
│   ├── solvers/        # e.g. replay: use a stored answer instead of generating
│   ├── scorers/        # grades, scoring instructions, completeness and additions scorers
│   ├── metrics/        # metrics that exclude N
│   ├── analysis/       # per-sample grade tables and per-model summaries
│   ├── maintainance/   # repairing existing logs in place (tested, tagged, backed up first)
│   └── experiments/    # run scripts, one per experiment
├── tests/              # mirrors src/; also checks every module imports and type-checks
├── explore/            # scripts for manual investigation; write no logs or results
├── logs/               # eval logs per experiment, not checked in
└── results/            # CSV tables per run script and git commit
```

## Prerequisites

```bash
brew install uv
brew install ollama
```

## Project setup

1. `uv sync` creates `.venv` and fetches Python 3.12 if needed.
2. Make sure Ollama is running.
3. Run `src/utils/pull_models.py` to download missing models (several GB each).

Check what's installed with `ollama list`, disk use with `du -sh ~/.ollama/models`.

## Running experiments

Experiments are configured in code, so every run is reproducible from git.
Each run script in `src/experiments/` writes its logs to
`logs/experiments/<experiment>/` and its result tables to
`results/<run script>/<commit>/`. Browse logs with
`uv run inspect view --log-dir logs/experiments`.

Ollama serves one request at a time, so run scripts keep Inspect serial (`max_tasks=1`, `max_samples=1`,
`max_connections=1`). Parallel requests only queue.

## Tests

pytest with PyHamcrest. Besides unit and integration tests, two tests guard the whole codebase: every module under
`src/` must import, and `ty` must report no type errors.

## Adding a model

1. Add it to the `Model` enum in `src/utils/models.py`, with a comment on maker, openness and licence
2. Run `src/utils/pull_models.py`
3. Quick check: `ollama run <tag> "hi"`
4. Use it in a run script as `Model.<NAME>`

## Hardware notes

MacBook Air M5, 24 GB unified memory. Models up to about 14B parameters run comfortably, around 20B is workable, 24B and
above is tight.