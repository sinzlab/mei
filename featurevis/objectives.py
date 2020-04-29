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


class RegularIntervalObjective(Objective):
    """Computes intervals in regular intervals during the optimization process.

    Attributes:
        interval: An integer greater than zero representing the number of optimization steps between the computation of
            the objective.
    """

    def __init__(self, interval: int):
        if interval <= 0:
            raise ValueError(f"Expected interval to be an integer greater than 0, got {interval}")
        self.interval = interval

    def __call__(self, current_state: State) -> Any:
        if current_state.i_iter % self.interval == 0:
            return self.compute(current_state)

    @abstractmethod
    def compute(self, current_state: State) -> Any:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({self.interval})"


class EvaluationObjective(RegularIntervalObjective):
    def compute(self, current_state: State) -> Any:
        return current_state.evaluation
