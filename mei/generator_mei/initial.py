from abc import ABC, abstractmethod

import torch


class InitialGuessCreator(ABC):
    """Implements the interface used to create an initial guess."""

    @abstractmethod
    def __call__(self, *shape) -> torch.Tensor:
        """Creates an initial guess from which to start the MEI optimization process given a shape."""


class RandomNormal(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        return torch.randn(*shape) * self.sigma + self.mu

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self.mu}, {self.sigma})"
