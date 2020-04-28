"""This module contains domain models."""

from __future__ import annotations
from typing import Any

from torch import Tensor


class Input:
    """Domain model representing the input to a model.

    Attributes:
        tensor: A PyTorch tensor containing floats.
    """

    def __init__(self, tensor: Tensor):
        """Initializes Input."""
        self.tensor = tensor
        self.tensor.requires_grad_()

    @property
    def gradient(self) -> Tensor:
        return self.tensor.grad

    @gradient.setter
    def gradient(self, value: Tensor):
        self.tensor.grad = value

    @property
    def data(self) -> Tensor:
        return self.tensor.data

    @data.setter
    def data(self, value: Tensor):
        self.tensor.data = value

    def extract(self) -> Tensor:
        """Extracts and returns the current tensor."""
        return self.tensor.detach().clone().cpu().squeeze()

    def clone(self) -> Input:
        """Returns a new instance of Input with a cloned tensor."""
        return Input(self.tensor.clone())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({repr(self.tensor)})"


class State:
    def __init__(
        self,
        i_iter: int,
        evaluation: float,
        input_: Tensor,
        transformed_input: Tensor,
        post_processed_input: Tensor,
        grad: Tensor,
        preconditioned_grad: Tensor,
        stopper_output: Any,
    ):
        self.i_iter = i_iter
        self.evaluation = evaluation
        self.input = input_
        self.transformed_input = transformed_input
        self.post_processed_input = post_processed_input
        self.grad = grad
        self.preconditioned_grad = preconditioned_grad
        self.stopper_output = stopper_output

    def __repr__(self) -> str:
        representation = self.__class__.__qualname__ + "("
        representation += ", ".join(
            repr(x)
            for x in (
                self.i_iter,
                self.evaluation,
                self.input,
                self.transformed_input,
                self.post_processed_input,
                self.grad,
                self.preconditioned_grad,
                self.stopper_output,
            )
        )
        representation += ")"
        return representation

    def to_dict(self):
        return dict(
            i_iter=self.i_iter,
            evaluation=self.evaluation,
            input_=self.input,
            transformed_input=self.transformed_input,
            post_processed_input=self.post_processed_input,
            grad=self.grad,
            preconditioned_grad=self.preconditioned_grad,
            stopper_output=self.stopper_output,
        )
