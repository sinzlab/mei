"""This module contains PyTorch modules used in the MEI optimization process."""

from typing import Dict, Any

import torch
from torch import Tensor
from torch.nn import Module, ModuleList


class EnsembleModel(Module):
    """A ensemble model consisting of several individual ensemble members.

    Attributes:
        *members: PyTorch modules representing the members of the ensemble.
    """

    _module_container_cls = ModuleList

    def __init__(self, *members: Module):
        """Initializes EnsembleModel."""
        super().__init__()
        self.members = self._module_container_cls(members)

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Calculates the forward pass through the ensemble.

        The input is passed through all individual members of the ensemble and their outputs are averaged.

        Args:
            x: A tensor representing the input to the ensemble.
            *args: Additional arguments will be passed to all ensemble members.
            **kwargs: Additional keyword arguments will be passed to all ensemble members.

        Returns:
            A tensor representing the ensemble's output.
        """
        outputs = [m(x, *args, **kwargs) for m in self.members]
        mean_output = torch.stack(outputs, dim=0).mean(dim=0)
        return mean_output

    def __repr__(self):
        return f"{self.__class__.__qualname__}({', '.join(m.__repr__() for m in self.members)})"


class ConstrainedOutputModel(Module):
    """A model that has its output constrained.

    Attributes:
        model: A PyTorch module.
        constraint: An integer representing the index of a neuron in the model's output. Only the value corresponding
            to that index will be returned.
        forward_kwargs: A dictionary containing keyword arguments that will be passed to the model every time it is
            called. Optional.
    """

    def __init__(self, model: Module, constraint: int, forward_kwargs: Dict[str, Any] = None):
        """Initializes ConstrainedOutputModel."""
        super().__init__()
        self.model = model
        self.constraint = constraint
        self.forward_kwargs = forward_kwargs if forward_kwargs else dict()

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Computes the constrained output of the model.

        Args:
            x: A tensor representing the input to the model.
            *args: Additional arguments will be passed to the model.
            **kwargs: Additional keyword arguments will be passed to the model.

        Returns:
            A tensor representing the constrained output of the model.
        """
        output = self.model(x, *args, **self.forward_kwargs, **kwargs)
        return output[:, self.constraint]

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self.model}, {self.constraint}, forward_kwargs={self.forward_kwargs})"
