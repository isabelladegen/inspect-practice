from os import path

from inspect_ai import eval_set
from inspect_ai.analysis import evals_df

from src.utils.config import EXP_LOG_DIR
from src.utils.models import Model
from src.tasks import hello_world

if __name__ == "__main__":
    temperature = 0.0  # most likely response
    seed = 42  # passed to Ollama for repeatable sampling

    TASKS = [hello_world()]
    MODELS = [Model.GEMMA3_12B, Model.OLMO2_13B, Model.PHI4_14B, Model.APERTUS_V11_4B]
    EXPERIMENT = "hello_world_v1"

    log_path = path.join(EXP_LOG_DIR, EXPERIMENT)
    eval_set(
        tasks=TASKS,
        model=MODELS,
        log_dir=log_path,
        temperature=temperature,
        seed=seed,
        max_tasks=1,  # one model at a time
        max_tokens=50,  # hello world needs ~5 tokens; stops runaway generation
        timeout=120,  # per request, fail rather than hang
        max_retries=0,
    )

    results = evals_df(log_path)  # data frame of run, one row per run
    print(results[["model", "status", "score_headline_value"]])
