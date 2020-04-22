"""This module contains domain models."""

from __future__ import annotations
from torch import Tensor


class Input:
    def __init__(self, tensor: Tensor):
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
        return self.tensor.detach().clone().cpu().squeeze()

    def clone(self) -> Input:
        return Input(self.tensor.clone())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({repr(self.tensor)})"
