"""Contains classes and functions related to optimizing an input to a function such that its value is maximized."""

from __future__ import annotations
from typing import Callable, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .domain import State
from .stoppers import OptimizationStopper
from .tracking import Tracker
from tqdm import tqdm
from itertools import count
from torch import nn


# noinspection PyUnusedLocal
def default_transform(mei: Tensor, iteration: int) -> Tensor:
    """Default transform used when no transform is provided to MEI."""
    return mei


# noinspection PyUnusedLocal
def default_regularization(mei: Tensor, iteration: int) -> Tensor:
    """Default regularization used when no regularization is provided to MEI."""
    return torch.tensor(0.0)


# noinspection PyUnusedLocal
def default_precondition(generator: nn.Module, iteration: int) -> Tensor:
    """Default preconditioning used when no preconditioning is provided to MEI."""
    return


# noinspection PyUnusedLocal
def default_postprocessing(generator: nn.Module, iteration: int) -> Tensor:
    """Default postprocessing function used when not postprocessing function is provided to MEI."""
    return


class MEI:
    """Wrapper around the function and the MEI tensor."""

    def __init__(
        self,
        func: Callable[[Tensor], Tensor],
        generator: nn.Module,
        optimizer: Optimizer,
        stopper: Callable,
        tracker: Callable,
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
        self.func = func
        self.generator = generator  # save initial state as a separate clone
        self.optimizer = optimizer
        self.transform = transform
        self.regularization = regularization
        self.precondition = precondition
        self.postprocessing = postprocessing
        self.stopper = stopper
        self.tracker = tracker
        self.i_iteration = 0
        self._transformed = None

    @property
    def transformed_input(self) -> Tensor:
        return self.transform(self.generator(), iteration=self.i_iteration)

    def evaluate(self, input=None) -> Tensor:
        """Evaluates the function on the current MEI."""
        return self.func(self.transformed_input if input is None else input)

    def step(self) -> State:
        """Performs an optimization step."""
        self.optimizer.zero_grad()
        current_input = self.generator()
        state = dict(i_iter=self.i_iteration, input_=current_input.data.cpu().clone())
        transformed_input = self.transform(current_input, iteration=self.i_iteration)
        state["transformed_input"] = transformed_input.data.cpu().clone()
        evaluation = self.func(transformed_input)
        state["evaluation"] = float(evaluation)  # extract scaler value
        reg_term = self.regularization(transformed_input, iteration=self.i_iteration)
        if hasattr(self.generator, "regularization"):
            reg_term += self.generator.regularization()
        state["reg_term"] = float(reg_term)
        (-evaluation + reg_term).backward()
        state["grad"] = 0
        self.precondition(self.generator, self.i_iteration)
        state["preconditioned_grad"] = 0
        self.optimizer.step()
        self.postprocessing(self.generator, self.i_iteration)
        state["post_processed_input"] = self.generator()
        self.i_iteration += 1
        return State.from_dict(state)

    def optimize(self):
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
        for _ in tqdm(count()):
            current_state = self.step()
            stop, output = self.stopper(current_state)
            current_state.stopper_output = output
            self.tracker.track(current_state)
            if stop:
                break
        return current_state.evaluation, current_state.post_processed_input

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}({self.func}, {self.generator}, {self.optimizer}, "
            f"transform={self.transform}, regularization={self.regularization}, precondition={self.precondition}, "
            f"postprocessing={self.postprocessing})"
        )


# def optimize(
#     mei: MEI, stopper: OptimizationStopper, tracker: Tracker
# ) -> Tuple[float, Tensor]:

#     for _ in tqdm(count()):
#         current_state = mei.step()
#         stop, output = stopper(current_state)
#         current_state.stopper_output = output
#         tracker.track(current_state)
#         if stop:
#             break
#     return current_state.evaluation, current_state.post_processed_input
