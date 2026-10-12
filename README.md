# inspect-practice

Practice repo for running Inspect evals against local models served by Ollama.

## Architecture

- **Eval side:** this repo. Python 3.12, managed by uv, with Inspect.
- **Model side:** Ollama, installed system-wide, running as a background
  service on `localhost:11434`. Inspect talks to it over HTTP.
- Models are listed in `models.txt` and pulled with `pull_models.sh`.
  Downloaded weights live in `~/.ollama/models`, outside this repo.

## Prerequirements

```bash
brew install uv
brew install ollama
brew services start ollama
```

## Project setup

```bash
uv sync                   # creates .venv and uv.lock
./pull_models.sh          # downloads models in models.txt (several GB each)
```

Check what's installed with `ollama list`.

## Running evals

```bash
uv run inspect eval tasks/<task>.py --model ollama/gemma3:12b
uv run inspect view       # browse results in the browser
```

Run one model per command. With 24 GB of memory, loading
several models at once slows everything down.

## Adding a model

1. Add its tag to `models.txt`
2. Run `./pull_models.sh`
3. Use it with `--model ollama/<tag>`

## Hardware notes

MacBook Air M5, 24 GB unified memory. Models up to about 14B parameters run comfortably, around 20B is workable, 24B and
above is tight.