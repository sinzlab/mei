from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Tuple

# Prevents circular import error
if TYPE_CHECKING:
    from torch import Tensor
    from torch.optim.optimizer import Optimizer

    from .stoppers import OptimizationStopper


def default_transform(mei, _i_iteration):
    """Default transform used when no transform is provided to MEI."""
    return mei


def default_regularization(_mei, _i_iteration):
    """Default regularization used when no regularization is provided to MEI."""
    return 0


def default_precondition(gradient, _i_iteration):
    """Default preconditioning used when no preconditioning is provided to MEI."""
    return gradient


def default_postprocess(mei, _i_iteration):
    """Default postprocessing function used when not postprocessing function is provided to MEI."""
    return mei


class MEI:
    """Wrapper around the function and the MEI tensor."""

    def __init__(
        self,
        func: Callable[[Tensor], Tensor],
        initial: Tensor,
        optimizer: Optimizer,
        transform: Callable[[Tensor, int], Tensor] = default_transform,
        regularization: Callable[[Tensor, int], Tensor] = default_regularization,
        precondition: Callable[[Tensor, int], Tensor] = default_precondition,
    ):
        """Initializes MEI.

        Args:
            func: A callable that will receive the to be optimized MEI tensor of floats as its only argument and that
                must return a tensor containing a single float.
            initial: A tensor containing floats representing the initial guess to start the optimization process
                from.
            optimizer: A PyTorch-style optimizer class.
            transform: A callable that will receive the current MEI and the index of the current iteration as inputs and
                that must return a transformed version of the current MEI. Optional.
            regularization: A callable that should have the current mei and the index of the current iteration as
                parameters and that should return a regularization term.
            precondition: A callable that should have the gradient of the MEI and the index of the current iteration as
                parameters and that should return a preconditioned gradient.
        """
        self.func = func
        self.initial = initial
        self.optimizer = optimizer
        self.transform = transform
        self.regularization = regularization
        self.precondition = precondition
        self.i_iteration = 0
        self._mei = self.initial
        self._mei.requires_grad_()
        self.__transformed_mei = None

    @property
    def _transformed_mei(self):
        if self.__transformed_mei is None:
            self.__transformed_mei = self.transform(self._mei, self.i_iteration)
        return self.__transformed_mei

    def evaluate(self) -> Tensor:
        """Evaluates the function on the current MEI."""
        return self.func(self._transformed_mei)

    def step(self) -> Tensor:
        """Performs an optimization step."""
        self.optimizer.zero_grad()
        evaluation = self.evaluate()
        reg_term = self.regularization(self._transformed_mei, self.i_iteration)
        (-evaluation + reg_term).backward()
        self._mei.grad = self.precondition(self._mei.grad, self.i_iteration)
        self.optimizer.step()
        self.__transformed_mei = None
        self.i_iteration += 1
        return evaluation

    @property
    def mei(self) -> Tensor:
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
    return evaluation.item(), mei.mei
