from os import path

from inspect_ai import eval_set

from utils.config import EXP_LOG_DIR
from utils.models import Model
from tasks.hello_world import hello_world

if __name__ == "__main__":
    temperature = 0.0  # most likely response
    seed = 42  # passed to Ollama for repeatable sampling

    TASKS = [hello_world()]
    MODELS = [Model.GEMMA3_12B, Model.OLMO2_13B, Model.PHI4_14B, Model.APERTUS_V11_4B]
    EXPERIMENT = "hello_world_v1"

    eval_set(
        tasks=TASKS,
        model=MODELS,
        log_dir=path.join(EXP_LOG_DIR, EXPERIMENT),
        temperature=temperature,
        seed=seed,
        max_tasks=1,      # one model at a time
        max_tokens=50,    # hello world needs ~5 tokens; stops runaway generation
        timeout=120,      # per request, fail rather than hang
        max_retries=0,
    )
