"""Explicit, request-local candidate selection; never a saved user preference."""
from contextlib import contextmanager
from contextvars import ContextVar

EXPERIMENTS = (
    "baseline",
    "action_first",
    "fight_choreography",
    "planning_thinking",
)
_experiment = ContextVar("promptbench_experiment", default="baseline")


def validate_experiment(value):
    if value not in EXPERIMENTS:
        raise ValueError(f"Unknown prompt-bench experiment: {value}")
    return value


def action_first_enabled():
    return _experiment.get() == "action_first"


def fight_choreography_enabled():
    return _experiment.get() == "fight_choreography"


def planning_thinking_enabled():
    return _experiment.get() == "planning_thinking"


@contextmanager
def experiment_context(value="baseline"):
    token = _experiment.set(validate_experiment(value))
    try:
        yield
    finally:
        _experiment.reset(token)
