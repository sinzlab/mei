"""This module contains domain models."""

from __future__ import annotations
from typing import Any, Dict, Optional

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
    def grad(self) -> Tensor:
        return self.tensor.grad

    @grad.setter
    def grad(self, value: Tensor):
        self.tensor.grad = value

    @property
    def cloned_grad(self) -> Tensor:
        """Returns a cloned CPU version of the gradient."""
        return self.grad.cpu().clone()

    @property
    def data(self) -> Tensor:
        return self.tensor.data

    @data.setter
    def data(self, value: Tensor):
        self.tensor.data = value

    @property
    def cloned_data(self) -> Tensor:
        """Returns a cloned CPU version of the data."""
        return self.data.cpu().clone()

    def clone(self) -> Input:
        """Returns a new instance of Input with a cloned tensor."""
        return Input(self.tensor.clone())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({repr(self.tensor)})"


class State:
    """The (current) state of the optimization process.

    Attributes:
        i_iter: An integer representing the index of the optimization step this state corresponds to.
        evaluation: A float representing the evaluation of the function in response to the current input.
        reg_term: A float representing the current regularization term added to the evaluation before the optimization
            step represented by this state was made. This value will be zero if no transformation is used.
        input_: A tensor representing the untransformed input to the function. This will be identical to the
            post-processed input from the last step for all steps except the first one.
        transformed_input: A tensor representing the transformed input to the function. This will be identical to the
            untransformed input if no transformation is used.
        post_processed_input: A tensor representing the post-processed input. This will be identical to the
            untransformed input if no post-processing is done.
        grad: A tensor representing the gradient.
        preconditioned_grad: A tensor representing the preconditioned gradient. This will be identical to the gradient
            if no preconditioning is done.
        stopper_output: An object returned by the stopper object. Optional.
    """

    def __init__(
        self,
        i_iter: int,
        evaluation: float,
        reg_term: float,
        input_: Tensor,
        transformed_input: Tensor,
        post_processed_input: Tensor,
        grad: Tensor,
        preconditioned_grad: Tensor,
        stopper_output: Optional[Any] = None,
    ):
        self.i_iter = i_iter
        self.evaluation = evaluation
        self.reg_term = reg_term
        self.input = input_
        self.transformed_input = transformed_input
        self.post_processed_input = post_processed_input
        self.grad = grad
        self.preconditioned_grad = preconditioned_grad
        self.stopper_output = stopper_output

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({', '.join(repr(v) for v in self.to_dict().values())})"

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the State."""
        return dict(
            i_iter=self.i_iter,
            evaluation=self.evaluation,
            reg_term=self.reg_term,
            input_=self.input,
            transformed_input=self.transformed_input,
            post_processed_input=self.post_processed_input,
            grad=self.grad,
            preconditioned_grad=self.preconditioned_grad,
            stopper_output=self.stopper_output,
        )

    def __eq__(self, other: State) -> bool:
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return self.to_dict() == other.to_dict()

    @classmethod
    def from_dict(cls, state: Dict[str, Any]):
        """Creates a new State object from a dictionary."""
        return cls(**state)
