import asyncio

from inspect_ai.core import ModelOutput, Target
from inspect_ai.model import ModelName
from inspect_ai.solver import TaskState

from src.experiments.grading_security_guide_same_answer_accross_models import fair_grading_message
from src.scorers.completeness_and_additions import completeness_of_several_answers, additions_of_several_answers
from src.tasks.replay import answers_by_item, answers_by_model
from src.utils.inspect_logs import successful_logs
from src.utils.models import Model

if __name__ == "__main__":
    item_id = 12
    grader = Model.PHI4_14B.as_inspect(temperature=0.0, max_tokens=2048, system_message=fair_grading_message())
    samples = answers_by_item(answers_by_model(successful_logs("security_guide_V2")), seed=1)
    sample = next(s for s in samples if s.id == item_id)

    target = Target(sample.target)
    state = TaskState(model=ModelName("mockllm/replay"), sample_id=sample.id, epoch=1, input=sample.input,
                      messages=[], target=target,
                      output=ModelOutput.from_content(model="replay", content=sample.metadata["answer"]),
                      metadata=sample.metadata)

    loop = asyncio.new_event_loop()
    for name, scorer in [("completeness", completeness_of_several_answers(grader=grader)),
                         ("additions", additions_of_several_answers(grader=grader))]:
        score = loop.run_until_complete(scorer(state, target))
        print(f"===== item {item_id}, {name} =====")
        print(score.explanation)
        print("----- grades -----")
        print(score.value)
    loop.close()