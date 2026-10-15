from os import path

from inspect_ai import eval_set
from inspect_ai.analysis import evals_df

from src.tasks.security_guide import security_guide
from src.utils.config import EXP_LOG_DIR
from src.utils.models import Model

if __name__ == "__main__":
    EXPERIMENT = "security_guide_V2"
    temperature = 0.0  # most likely response
    seed = 42  # passed to Ollama for repeatable sampling
    log_path = path.join(EXP_LOG_DIR, EXPERIMENT)

    security_sys_msg = """Your knowledge of security is being assessed. Respond to the following user questions 
    concisely but precisely, ideally in a few words.  
    Your answers will be assessed by a human security expert."""

    TASKS = [security_guide(s_message=security_sys_msg,
                            grader=Model.PHI4_14B,
                            grader_max_tokens=512,
                            grader_temp=temperature, )]
    MODELS : list[str] = [Model.GEMMA3_12B, Model.OLMO2_13B, Model.PHI4_14B, Model.APERTUS_V11_4B]

    eval_set(
        tasks=TASKS,
        model=MODELS,
        log_dir=log_path,
        temperature=0.0,
        seed=42,
        max_tasks=1,  # keep: one model loaded at a time
        max_connections=1,  # Ollama is effectively serial; avoids queue timeouts
        max_tokens=256,  # small models ignore "a few words"; leave headroom
        timeout=180,
        max_retries=1,
        retry_attempts=1,  # cap eval_set's task-level retries
    )

    results = evals_df(log_path)  # data frame of run, one row per run
    print(results[["model", "status", "score_headline_value"]])
