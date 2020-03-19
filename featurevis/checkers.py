"""Contains callable classes used to check if the MEI optimization process has reached an acceptable result."""

from __future__ import annotations
from abc import ABC, abstractmethod

from torch import Tensor

from .optimization import MEI


class OptimizationChecker(ABC):
    """Implements the interface used to check if the MEI optimization process has reached an acceptable result."""

    @abstractmethod
    def __call__(self, mei: MEI, evaluation: Tensor) -> bool:
        """Should return "True" if the MEI optimization process has reached an acceptable result."""


class NumIterations(OptimizationChecker):
    """Callable that stops the optimization process after a specified number of steps."""

    def __init__(self, num_iterations):
        """Initializes NumIterations.

        Args:
            num_iterations: The number of optimization steps before the process is stopped.
        """
        self.num_iterations = num_iterations
        self._current_iteration = 0

    def __call__(self, mei: MEI, evaluation: float) -> bool:
        """Stops the optimization process after a set number of steps by returning True."""
        if self._current_iteration == self.num_iterations:
            return True
        else:
            self._current_iteration += 1
            return False

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self.num_iterations})"
