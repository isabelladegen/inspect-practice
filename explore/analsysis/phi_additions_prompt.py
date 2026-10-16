from src.scorers.completeness_and_additions import additions, additions_of_several_answers
from src.utils.inspect_logs import logged_samples, successful_logs
from src.utils.inspect_registery import registered_name


def text(message) -> str:
    """The text of a logged chat message, read back as an object or as a plain dict."""
    content = message["content"] if isinstance(message, dict) else message.content
    if isinstance(content, str):
        return content
    parts = [part if isinstance(part, dict) else part.model_dump() for part in content]
    return "\n".join(part["text"] for part in parts if part.get("type") == "text")


def print_grading(title: str, score) -> None:
    prompt = score.metadata["grading"][0]  # the user message model_graded_fact sent
    print(f"\n========== {title} ==========")
    print("----- prompt -----")
    print(text(prompt))
    print("----- reply -----")
    print(score.explanation)
    print(f"----- value: {score.value}")


if __name__ == "__main__":
    item_ids = [9, 16]
    grader = "ollama/phi4:14b"

    single = {(s.id, log.eval.model): s.scores[registered_name(additions)]
              for log in successful_logs("security_guide_V2_rescore_completeness_and_additions_V3")
              for s in logged_samples(log) if s.id in item_ids}
    joint = {s.id: s for log in successful_logs("security_guide_V2_rescore_several_answers_V1")
             if log.eval.model == grader
             for s in logged_samples(log) if s.id in item_ids}

    for item_id in item_ids:
        for (scored_item, model), score in sorted(single.items()):
            if scored_item == item_id:
                print_grading(f"item {item_id}, single, answer of {model}", score)
        sample = joint[item_id]
        print(f"\n----- not sent to the grader: label to model -----\n{sample.metadata['models']}")
        print_grading(f"item {item_id}, joint, all answers",
                      sample.scores[registered_name(additions_of_several_answers)])