from abc import ABC, abstractmethod

from torch import Tensor, randn


class InitialGuessCreator(ABC):
    """Implements the interface used to create an initial guess."""

    @abstractmethod
    def __call__(self, *shape) -> Tensor:
        """Creates an initial guess from which to start the MEI optimization process given a shape."""


class RandomNormal(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        return self._create_random_tensor(*shape)

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"
