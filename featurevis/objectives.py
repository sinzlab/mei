from abc import ABC, abstractmethod
from typing import Any

from .domain import State


class Objective(ABC):
    """Abstract base class all objectives must inherit from."""

    def __call__(self, current_state: State) -> Any:
        return self.compute(current_state)

    @abstractmethod
    def compute(self, current_state: State) -> Any:
        """Returns an object computed from the current state of the optimization process."""


class Evaluation(Objective):
    def compute(self, current_state: State) -> Any:
        return current_state.evaluation
