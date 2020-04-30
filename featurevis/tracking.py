"""Contains classes and functions related to tracking objectives during the optimization process."""

from typing import Callable

from .domain import State


class Tracker:
    """Tracks the MEI optimization process.

    Attributes:
        objectives: A number of objectives to be tracked. The must be passed as keyword arguments and they must be
            callable.
        log: A dictionary where each key is the name of a tracked objective and the values are dictionaries containing
            two keys: "times" and "values". The "times" key corresponds to a list of integers representing the indexes
            of the optimization steps in which the objective in question was tracked. The "values" key corresponds to a
            list of objects representing the corresponding results.
    """

    def __init__(self, **objectives: Callable):
        """Initializes Tracker."""
        self.objectives = objectives
        self.log = {n: dict(times=list(), values=list()) for n in objectives}

    def track(self, current_state: State) -> None:
        """Passes the current state of the MEI optimization process to each objective and logs the result."""
        for name, objective in self.objectives.items():
            if (result := objective(current_state)) is not None:
                self.log[name]["times"].append(current_state.i_iter)
                self.log[name]["values"].append(result)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({', '.join(self.objectives)})"
