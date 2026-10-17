import asyncio

from inspect_ai.model import ModelName, ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from src.experiments.grading_security_guide_same_answer_accross_models import fair_grading_message
from src.scorers.completeness_and_additions import additions, additions_of_several_answers
from src.tasks.replay import answers_by_item, answers_by_model
from src.utils.inspect_logs import successful_logs
from src.utils.models import Model


def state_for(item_id, question, key, answer: str, metadata: dict) -> TaskState:
    """A task state as the replay solver leaves it: the stored answer as the output."""
    return TaskState(model=ModelName("mockllm/replay"), sample_id=item_id, epoch=1, input=question,
                     messages=[], target=Target(key),
                     output=ModelOutput.from_content(model="replay", content=answer), metadata=metadata)


if __name__ == "__main__":
    # item_ids = [9, 16]
    item_ids = [9, 12, 14, 16]
    grader = Model.PHI4_14B
    answers = answers_by_model(successful_logs("security_guide_V2"))
    items = {s.id: s for s in answers_by_item(answers, seed=112) if s.id in item_ids}

    single = additions(grader=grader, max_tokens=1024, temp=0.0)
    joint = additions_of_several_answers(
        grader=grader.as_inspect(temperature=0.0, max_tokens=2048, system_message=fair_grading_message()))

    loop = asyncio.new_event_loop()
    for item_id in item_ids:
        for model, model_answers in answers.items():
            a = next(a for a in model_answers if a.item_id == item_id)
            score = loop.run_until_complete(single(state_for(item_id, a.question, a.key, a.answer, {}),
                                                   Target(a.key)))
            print(f"\n===== item {item_id}, single, {model} =====\n{score.explanation}"
                  f"\n----- grade: {score.value}")

        sample = items[item_id]
        score = loop.run_until_complete(joint(state_for(item_id, sample.input, sample.target,
                                                        sample.metadata["answer"], sample.metadata),
                                              Target(sample.target)))
        print(f"\n===== item {item_id}, joint =====\n----- not sent: {sample.metadata['models']}"
              f"\n{score.explanation}\n----- grades: {score.value}")
    loop.close()