"""This module contains domain models."""

from __future__ import annotations
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
