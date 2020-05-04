"""Contains callable classes representing objectives that can be tracked during the optimization process.

All objectives must be subclasses of the ABC called "Objective" and implement the "compute" method. Said method will be
called, with an object encapsulating the current state of the optimization process passed as the only argument, after
each optimization step (or in regular user-defined intervals). In response to being called the "compute" method should
return a result object computed based on the received state.
"""

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
    """Computes objectives in regular intervals during the optimization process.

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
    """Objective used to track the function evaluation during the optimization process."""

    def compute(self, current_state: State) -> Any:
        return current_state.evaluation


class PostProcessedInputObjective(RegularIntervalObjective):
    """Objective used to track the post-processed input to the function during the optimization process"""

    def compute(self, current_state: State) -> Any:
        return current_state.post_processed_input
