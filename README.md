# inspect-practice

Practice repo for running Inspect evals against local models served by Ollama.

## Architecture

- **Eval side:** this repo. Python 3.12, managed by uv, with Inspect.
- **Model side:** Ollama, installed system-wide, started on demand and
  listening on `localhost:11434`. Inspect talks to it over HTTP.
- Models are defined once in `utils/models.py` and pulled with
  `utils/pull_models.py`. Weights live in `~/.ollama/models`, outside this repo.

## Layout

```
inspect-practice/
├── pyproject.toml / uv.lock
├── utils/
│   ├── config.py         # project paths (root, log folders)
│   ├── models.py         # Model enum: single source of truth for model tags
│   └── pull_models.py    # pulls every model in the enum, skips installed ones
├── tasks/                # Inspect tasks
├── experiments/          # run scripts, one per experiment
└── logs/                 # eval and pull logs, gitignored
```

## Prerequisites

```bash
brew install uv
brew install ollama
```

## Project setup

```bash
uv sync                              # creates .venv and uv.lock, fetches Python 3.12 if needed
ollama serve                         # in a separate terminal, Ctrl-C to stop
uv run python -m utils.pull_models   # downloads missing models (several GB each)
```

Check what's installed with `ollama list`, disk use with `du -sh ~/.ollama/models`.

## Running evals

Experiments are run from code, so every run is reproducible from git:

```bash
uv run python -m experiments.run_hello_world
uv run inspect view --log-dir logs/experiments   # browse results
```

Run scripts set `max_tasks=1` so only one model is in memory at a time,
plus `max_tokens` and `timeout` so a misbehaving model fails rather than hangs.

## Adding a model

1. Add it to the `Model` enum in `utils/models.py`, with a comment on maker,
   openness and licence
2. Run `uv run python -m utils.pull_models`
3. Quick check: `ollama run <tag> "hi"`
4. Use it in a run script as `Model.<NAME>`

## Hardware notes

MacBook Air M5, 24 GB unified memory. Models up to about 14B parameters run
comfortably, around 20B is workable, 24B and above is tight.