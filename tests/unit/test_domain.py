from unittest.mock import MagicMock

import pytest
from torch import Tensor

from mei import domain


class TestInput:
    @pytest.fixture
    def input_(self, tensor):
        return domain.Input(tensor)

    @pytest.fixture
    def tensor(self, grad, data):
        tensor = MagicMock(name="tensor", spec=Tensor)
        tensor.grad = grad
        tensor.data = data
        tensor.__repr__ = MagicMock(name="repr", return_value="repr")
        return tensor

    @pytest.fixture
    def grad(self):
        grad = MagicMock(name="grad", spec=Tensor)
        grad.cpu.return_value.clone.return_value = "cloned_grad"
        return grad

    @pytest.fixture
    def data(self):
        data = MagicMock(name="data", spec=Tensor)
        data.cpu.return_value.clone.return_value = "cloned_data"
        return data

    def test_init(self, input_, tensor):
        assert input_.tensor is tensor

    def test_if_gradient_gets_enable_on_provided_tensor(self, input_, tensor):
        tensor.requires_grad_.assert_called_once_with()

    def test_gradient_property(self, input_, tensor, grad):
        assert input_.grad is grad

    def test_gradient_setter(self, input_, tensor):
        input_.grad = "new_grad"
        assert tensor.grad == "new_grad"

    def test_if_cloned_gradient_property_calls_cpu_method_correctly(self, input_, grad):
        _ = input_.cloned_grad
        grad.cpu.assert_called_once_with()

    def test_if_cloned_gradient_property_calls_clone_method_correctly(self, input_, grad):
        _ = input_.cloned_grad
        grad.cpu.return_value.clone.assert_called_once_with()

    def test_if_cloned_gradient_property_returns_correct_value(self, input_):
        assert input_.cloned_grad == "cloned_grad"

    def test_data_property(self, input_, tensor, data):
        assert input_.data == data

    def test_data_setter(self, input_, tensor):
        input_.data = "new_data"
        assert tensor.data == "new_data"

    def test_if_cloned_data_property_calls_cpu_method_correctly(self, input_, data):
        _ = input_.cloned_data
        data.cpu.assert_called_once_with()

    def test_if_cloned_data_property_calls_clone_method_correctly(self, input_, data):
        _ = input_.cloned_data
        data.cpu.return_value.clone.assert_called_once_with()

    def test_if_cloned_data_property_returns_correct_value(self, input_):
        assert input_.cloned_data == "cloned_data"

    def test_if_tensor_is_cloned_when_cloned(self, input_, tensor):
        input_.clone()
        tensor.clone.assert_called_once_with()

    def test_if_clone_returns_a_new_input_instance(self, input_):
        cloned = input_.clone()
        assert isinstance(cloned, domain.Input) and cloned is not input_

    def test_repr(self, input_, tensor):
        assert repr(input_) == f"Input({repr(tensor)})"


class TestState:
    @pytest.fixture
    def state_data(self):
        input_ = MagicMock(name="input", spec=Tensor)
        input_.__repr__ = MagicMock(return_value="input")
        transformed_input = MagicMock(name="transformed_input", spec=Tensor)
        transformed_input.__repr__ = MagicMock(return_value="transformed_input")
        post_processed_input = MagicMock(name="post_processed_input", spec=Tensor)
        post_processed_input.__repr__ = MagicMock(return_value="post_processed_input")
        grad = MagicMock(name="grad", spec=Tensor)
        grad.__repr__ = MagicMock(return_value="grad")
        preconditioned_grad = MagicMock(name="preconditioned_grad", spec=Tensor)
        preconditioned_grad.__repr__ = MagicMock(return_value="preconditioned_grad")
        stopper_output = MagicMock(name="stopper_output", spec=Tensor)
        stopper_output.__repr__ = MagicMock(return_value="stopper_output")
        state_data = dict(
            i_iter=10,
            evaluation=3.4,
            reg_term=5.1,
            input_=input_,
            transformed_input=transformed_input,
            post_processed_input=post_processed_input,
            grad=grad,
            preconditioned_grad=preconditioned_grad,
            stopper_output=stopper_output,
        )
        return state_data

    def test_init(self, state_data):
        state = domain.State(**state_data)
        assert (
            state.i_iter is state_data["i_iter"]
            and state.evaluation is state_data["evaluation"]
            and state.reg_term is state_data["reg_term"]
            and state.input is state_data["input_"]
            and state.transformed_input is state_data["transformed_input"]
            and state.post_processed_input is state_data["post_processed_input"]
            and state.grad is state_data["grad"]
            and state.preconditioned_grad is state_data["preconditioned_grad"]
            and state.stopper_output is state_data["stopper_output"]
        )

    def test_if_stopper_output_is_optional(self, state_data):
        del state_data["stopper_output"]
        state = domain.State(**state_data)
        assert state.stopper_output is None

    def test_repr(self, state_data):
        state = domain.State(**state_data)
        assert (
            repr(state) == "State(10, 3.4, 5.1, input, transformed_input, "
            "post_processed_input, grad, preconditioned_grad, stopper_output)"
        )

    def test_to_dict(self, state_data):
        assert domain.State(**state_data).to_dict() == state_data

    def test_if_equality_raises_not_implemented_error_if_other_is_not_same_class(self, state_data):
        with pytest.raises(NotImplementedError):
            _ = domain.State(**state_data) == "not_a_state"

    def test_if_equality_returns_true_if_self_and_other_contain_same_data(self, state_data):
        assert domain.State(**state_data) == domain.State(**state_data)

    def test_if_equality_returns_false_if_self_and_other_contain_different_data(self, state_data):
        state = domain.State(**state_data)
        state.i_iter = 11
        assert not state == domain.State(**state_data)

    def test_from_dict(self, state_data):
        assert domain.State.from_dict(state_data) == domain.State(**state_data)
