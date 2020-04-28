from typing import Callable

from .domain import State


class Tracker:
    """Tracks the MEI optimization process.

    Attributes:
        objectives: A number of objectives to be tracked. The must be passed as keyword arguments and they must be
            callable.
        log: A dictionary where each key is the name of a tracked objective and the values are lists containing the data
            returned by the corresponding objective across time.
    """

    def __init__(self, **objectives: Callable):
        """Initializes Tracker."""
        self.objectives = objectives
        self.log = {n: list() for n in objectives}

    def track(self, current_state: State) -> None:
        """Passes the current state of the MEI optimization process to each objective and logs the result."""
        for name, objective in self.objectives.items():
            self.log[name].append(objective(current_state))

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({', '.join(self.objectives)})"
