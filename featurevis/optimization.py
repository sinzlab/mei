from __future__ import annotations
from typing import TYPE_CHECKING, Callable

# Prevents circular import error
if TYPE_CHECKING:
    from torch import Tensor
    from torch.optim.optimizer import Optimizer

    from .checkers import OptimizationChecker


class MEI:
    """Wrapper around the function and the MEI tensor."""

    def __init__(self, func: Callable[[Tensor], Tensor], initial_guess: Tensor):
        """Initializes MEI.

        Args:
            func: A callable that will receive the to be optimized MEI tensor of floats as its only argument and that
                must return a tensor containing a single float.
            initial_guess: A tensor containing floats representing the initial guess to start the optimization process
                from.
        """
        self.func = func
        self.initial_guess = initial_guess
        self._mei = self._initialize_mei()

    def _initialize_mei(self) -> Tensor:
        """Detaches and clones the initial guess and enables the gradient on the cloned tensor."""
        mei = self.initial_guess.detach().clone()
        mei.requires_grad_()
        return mei

    def evaluate(self) -> Tensor:
        """Evaluates the current MEI on the callable and returns the result."""
        return self.func(self._mei)

    def __call__(self) -> Tensor:
        """Detaches and clones the current MEI and returns it."""
        return self._mei.detach().clone()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({self.func}, {self.initial_guess})"


def optimize(mei: MEI, optimizer: Optimizer, optimized: OptimizationChecker) -> Tensor:
    """Optimizes the input to a given function such that it maximizes said function using gradient ascent.

    Args:
        mei: An instance of the to be optimized MEI.
        optimizer: A PyTorch-style optimizer class.
        optimized: A subclass of "OptimizationChecker" used to stop the optimization process.

    Returns:
        A tensor of floats having the same shape as "initial_guess" representing the input that maximizes the function.
    """
    evaluation = mei.evaluate()
    while not optimized(mei, evaluation):
        optimizer.zero_grad()
        evaluation = mei.evaluate()
        (-evaluation).backward()
        optimizer.step()
    return mei()