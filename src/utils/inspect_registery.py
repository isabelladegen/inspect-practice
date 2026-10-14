from inspect_ai._util.registry import registry_info


def registered_name(obj) -> str:
    """The name Inspect registered for `obj` and writes into logs, without package prefix."""
    return registry_info(obj).name.split("/")[-1]


def scorer_metrics(scorer) -> list:
    """The metrics declared in `scorer`'s @scorer decorator."""
    metrics = registry_info(scorer).metadata["metrics"]
    if not isinstance(metrics, list):
        raise ValueError(f"expected a list of metrics for scorer {registered_name(scorer)}, "
                         f"got {type(metrics).__name__}")
    return metrics