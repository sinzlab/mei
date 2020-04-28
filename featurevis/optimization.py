from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Tuple
from dataclasses import dataclass

from .domain import State

# Prevents circular import error
if TYPE_CHECKING:
    from torch import Tensor
    from torch.optim.optimizer import Optimizer

    from .stoppers import OptimizationStopper
    from .domain import Input
    from .tracking import Tracker


@dataclass
class Gradient:
    regular: Tensor
    preconditioned: Tensor


def default_transform(mei: Tensor, _i_iteration: int) -> Tensor:
    """Default transform used when no transform is provided to MEI."""
    return mei


def default_regularization(_mei: Tensor, _i_iteration: int) -> 0:
    """Default regularization used when no regularization is provided to MEI."""
    return 0


def default_precondition(gradient: Tensor, _i_iteration: int) -> Tensor:
    """Default preconditioning used when no preconditioning is provided to MEI."""
    return gradient


def default_postprocessing(mei: Tensor, _i_iteration: int) -> Tensor:
    """Default postprocessing function used when not postprocessing function is provided to MEI."""
    return mei


class MEI:
    """Wrapper around the function and the MEI tensor."""

    def __init__(
        self,
        func: Callable[[Tensor], Tensor],
        initial: Input,
        optimizer: Optimizer,
        transform: Callable[[Tensor, int], Tensor] = default_transform,
        regularization: Callable[[Tensor, int], Tensor] = default_regularization,
        precondition: Callable[[Tensor, int], Tensor] = default_precondition,
        postprocessing: Callable[[Tensor, int], Tensor] = default_postprocessing,
    ):
        """Initializes MEI.

        Args:
            func: A callable that will receive the to be optimized MEI tensor of floats as its only argument and that
                must return a tensor containing a single float.
            initial: An instance of "Input" initialized with the tensor from which the optimization process will start.
            optimizer: A PyTorch-style optimizer class.
            transform: A callable that will receive the current MEI and the index of the current iteration as inputs and
                that must return a transformed version of the current MEI. Optional.
            regularization: A callable that should have the current mei and the index of the current iteration as
                parameters and that should return a regularization term.
            precondition: A callable that should have the gradient of the MEI and the index of the current iteration as
                parameters and that should return a preconditioned gradient.
            postprocessing: A callable that should have the current MEI and the index of the current iteration as
                parameters and that should return a post-processed MEI. The operation performed by this callable on the
                MEI has no influence on its gradient.
        """
        self.func = func
        self.initial = initial.clone()
        self.optimizer = optimizer
        self.transform = transform
        self.regularization = regularization
        self.precondition = precondition
        self.postprocessing = postprocessing
        self.i_iteration = 0
        self._current_input = initial
        self.__transformed_input = None

    @property
    def _transformed_input(self) -> Tensor:
        if self.__transformed_input is None:
            self.__transformed_input = self.transform(self._current_input.tensor, self.i_iteration)
        return self.__transformed_input

    def evaluate(self) -> Tensor:
        """Evaluates the function on the current MEI."""
        return self.func(self._transformed_input)

    def step(self) -> Tensor:
        """Performs an optimization step."""
        self.optimizer.zero_grad()
        evaluation = self.evaluate()
        reg_term = self.regularization(self._transformed_input, self.i_iteration)
        (-evaluation + reg_term).backward()
        if self._current_input.gradient is None:
            raise RuntimeError("Gradient did not reach MEI")
        self._current_input.gradient = self.precondition(self._current_input.gradient, self.i_iteration)
        self.optimizer.step()
        self._current_input.data = self.postprocessing(self._current_input.data, self.i_iteration)
        self.__transformed_input = None
        self.i_iteration += 1
        return evaluation

    @property
    def current_input(self) -> Tensor:
        """Detaches the current MEI and returns it."""
        return self._current_input.extract()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}({self.func}, {self.initial}, {self.optimizer}, "
            f"transform={self.transform}, regularization={self.regularization}, precondition={self.precondition}, "
            f"postprocessing={self.postprocessing})"
        )


def optimize(mei: MEI, stopper: OptimizationStopper, tracker: Tracker, state_cls=State) -> Tuple[float, Tensor]:
    """Optimizes the input to a given function such that it maximizes said function using gradient ascent.

    Args:
        mei: An instance of the to be optimized MEI.
        stopper: A subclass of "OptimizationStopper" used to stop the optimization process.
        tracker: A tracker object used to track pre-defined objectives during the optimization process. The current
            state of the optimization process is passed to the "track" method of the object in each iteration.
        state_cls: For testing purposes.

    Returns:
        A float representing the final evaluation and a tensor of floats having the same shape as "initial_guess"
        representing the input that maximizes the function.
    """
    evaluation = mei.evaluate()
    while True:
        stop, output = stopper(mei, evaluation)
        if stop:
            break
        evaluation, state = mei.step()
        state["stopper_output"] = output
        tracker.track(state_cls.from_dict(state))
    return evaluation.item(), mei.current_input
