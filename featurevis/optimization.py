from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Tuple

# Prevents circular import error
if TYPE_CHECKING:
    from torch import Tensor
    from torch.optim.optimizer import Optimizer

    from .stoppers import OptimizationStopper


class MEI:
    """Wrapper around the function and the MEI tensor."""

    def __init__(
        self,
        func: Callable[[Tensor], Tensor],
        initial: Tensor,
        optimizer: Optimizer,
        transform: Callable[[Tensor], Tensor] = None,
    ):
        """Initializes MEI.

        Args:
            func: A callable that will receive the to be optimized MEI tensor of floats as its only argument and that
                must return a tensor containing a single float.
            initial: A tensor containing floats representing the initial guess to start the optimization process
                from.
            optimizer: A PyTorch-style optimizer class.
            transform: A callable that will receive the current MEI and the index of the current iteration as inputs and
                that must return a transformed version of the current MEI.
        """
        self.func = func
        self.initial = initial
        self.optimizer = optimizer
        self.transform = self._initialize_transform(transform)
        self.i_iteration = 0
        self._mei = self.initial
        self._mei.requires_grad_()

    @staticmethod
    def _initialize_transform(transform):
        def identity(mei, **_kwargs):
            return mei

        if not transform:
            return identity
        else:
            return transform

    def evaluate(self) -> Tensor:
        """Evaluates the function on the current MEI."""
        return self.func(self.transform(self._mei, i_iteration=self.i_iteration))

    def step(self) -> Tensor:
        """Performs an optimization step."""
        self.optimizer.zero_grad()
        evaluation = self.evaluate()
        (-evaluation).backward()
        self.optimizer.step()
        self.i_iteration += 1
        return evaluation

    def get_mei(self) -> Tensor:
        """Detaches the current MEI and returns it."""
        return self._mei.detach().squeeze().cpu()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({self.func}, {self.initial})"


def optimize(mei: MEI, optimized: OptimizationStopper) -> Tuple[float, Tensor]:
    """Optimizes the input to a given function such that it maximizes said function using gradient ascent.

    Args:
        mei: An instance of the to be optimized MEI.
        optimized: A subclass of "OptimizationStopper" used to stop the optimization process.

    Returns:
        A float representing the final evaluation and a tensor of floats having the same shape as "initial_guess"
        representing the input that maximizes the function.
    """
    evaluation = mei.evaluate()
    while not optimized(mei, evaluation):
        evaluation = mei.step()
    return evaluation.item(), mei.get_mei()
