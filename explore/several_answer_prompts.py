from inspect_ai.core import ModelOutput
from inspect_ai.scorer._model import model_scoring_prompt

from src.experiments.grading_security_guide_same_answer_accross_models import fair_grading_message
from src.scorers.completeness_and_additions import several_facts_template, completeness_scoring, additions_scoring
from src.tasks.replay import answers_by_item, answers_by_model
from src.utils.inspect_logs import successful_logs

if __name__ == "__main__":
    item_id = 12
    samples = answers_by_item(answers_by_model(successful_logs("security_guide_V2")), seed=1)
    sample = next(s for s in samples if s.id == item_id)

    question = (sample.input if isinstance(sample.input, str)
                else next(m.text for m in reversed(sample.input) if m.role == "user"))
    for name, instructions in [("completeness", completeness_scoring()), ("additions", additions_scoring())]:
        prompt = model_scoring_prompt(
            template=several_facts_template(),
            question=question,
            output=ModelOutput.from_content(model="replay", content=sample.metadata["answer"]),
            criterion=sample.target,
            instructions=instructions.prompt_for_several_answers(),
            metadata={},
        )
        print(f"===== item {item_id}, {name} =====")
        print("----- system -----")
        print(fair_grading_message())
        print("----- user -----")
        print(prompt.text)

    print(f"----- not sent to the grader: label to model -----\n{sample.metadata['models']}")
    print(f"labels: {sample.metadata['models']}")