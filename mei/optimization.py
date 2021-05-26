"""Contains classes and functions related to optimizing an input to a function such that its value is maximized."""

from __future__ import annotations
from typing import Callable, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .domain import Input, State
from .stoppers import OptimizationStopper
from .tracking import Tracker


# noinspection PyUnusedLocal
def default_transform(mei: Tensor, i_iteration: int) -> Tensor:
    """Default transform used when no transform is provided to MEI."""
    return mei


# noinspection PyUnusedLocal
def default_regularization(mei: Tensor, i_iteration: int) -> Tensor:
    """Default regularization used when no regularization is provided to MEI."""
    return torch.tensor(0.0)


# noinspection PyUnusedLocal
def default_precondition(grad: Tensor, i_iteration: int) -> Tensor:
    """Default preconditioning used when no preconditioning is provided to MEI."""
    return grad


# noinspection PyUnusedLocal
def default_postprocessing(mei: Tensor, i_iteration: int) -> Tensor:
    """Default postprocessing function used when not postprocessing function is provided to MEI."""
    return mei


class MEI:
    """Wrapper around the function and the MEI tensor."""

    input_cls = Input
    state_cls = State

    def __init__(
        self,
        func: Callable[[Tensor], Tensor],
        initial: Tensor,
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
            initial: A tensor from which the optimization process will start.
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
        initial = self.input_cls(initial)
        self.func = func
        self.initial = initial.clone()
        self.optimizer = optimizer
        self.transform = transform
        self.regularization = regularization
        self.precondition = precondition
        self.postprocessing = postprocessing
        self.i_iteration = 0
        self._current_input = initial
        self._transformed = None

    @property
    def _transformed_input(self) -> Tensor:
        if self._transformed is None:
            self._transformed = self.transform(
                self._current_input.tensor, self.i_iteration
            )
        return self._transformed

    def evaluate(self) -> Tensor:
        """Evaluates the function on the current MEI."""
        return self.func(self._transformed_input)

    def step(self) -> State:
        """Performs an optimization step."""
        state = dict(i_iter=self.i_iteration, input_=self._current_input.cloned_data)
        self.optimizer.zero_grad()
        evaluation = self.evaluate()
        state["evaluation"] = evaluation.item()
        reg_term = self.regularization(self._transformed_input, self.i_iteration)
        state["reg_term"] = reg_term.item()
        state["transformed_input"] = self._transformed_input.data.cpu().clone()
        (-evaluation + reg_term).backward()
        if self._current_input.grad is None:
            raise RuntimeError("Gradient did not reach MEI")
        state["grad"] = self._current_input.cloned_grad
        self._current_input.grad = self.precondition(
            self._current_input.grad, self.i_iteration
        )
        state["preconditioned_grad"] = self._current_input.cloned_grad
        self.optimizer.step()
        self._current_input.data = self.postprocessing(
            self._current_input.data, self.i_iteration
        )
        state["post_processed_input"] = self._current_input.cloned_data
        self._transformed = None
        self.i_iteration += 1
        return self.state_cls.from_dict(state)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}({self.func}, {self.initial}, {self.optimizer}, "
            f"transform={self.transform}, regularization={self.regularization}, precondition={self.precondition}, "
            f"postprocessing={self.postprocessing})"
        )


def optimize(
    mei: MEI, stopper: OptimizationStopper, tracker: Tracker
) -> Tuple[float, Tensor]:
    """Optimizes the input to a given function such that it maximizes said function using gradient ascent.

    Args:
        mei: An instance of the to be optimized MEI.
        stopper: A subclass of "OptimizationStopper" used to stop the optimization process.
        tracker: A tracker object used to track pre-defined objectives during the optimization process. The current
            state of the optimization process is passed to the "track" method of the object in each iteration.

    Returns:
        A float representing the final evaluation and a tensor of floats having the same shape as "initial_guess"
        representing the input that maximizes the function.
    """
    while True:
        current_state = mei.step()
        stop, output = stopper(current_state)
        current_state.stopper_output = output
        tracker.track(current_state)
        if stop:
            break
    return current_state.evaluation, current_state.post_processed_input
