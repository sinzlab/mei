from unittest.mock import MagicMock, PropertyMock, call
from functools import partial

import pytest
from torch import Tensor

from mei import optimization
from mei.domain import Input, State


@pytest.fixture
def mei(
    input_cls, state_cls, func, initial_input_tensor, optimizer, transform, regularization, precondition, postprocessing
):
    optimization.MEI.input_cls = input_cls
    optimization.MEI.state_cls = state_cls
    return partial(
        optimization.MEI,
        func,
        initial_input_tensor,
        optimizer,
        transform=transform,
        regularization=regularization,
        precondition=precondition,
        postprocessing=postprocessing,
    )


@pytest.fixture
def input_cls(initial_input):
    return MagicMock(name="input_cls", spec=Input, return_value=initial_input)


@pytest.fixture
def state_cls():
    state_cls = MagicMock(name="state_cls", spec=State)
    state_cls.from_dict.return_value = "state_instance"
    return state_cls


@pytest.fixture
def func(evaluation):
    func = MagicMock(return_value=evaluation)
    representation = MagicMock(return_value="func")
    func.__repr__ = representation
    return func


@pytest.fixture
def initial_input_tensor():
    return MagicMock(name="initial_input_tensor", spec=Tensor)


@pytest.fixture
def evaluation(negated_evaluation):
    evaluation = MagicMock(name="evaluation")
    evaluation.__neg__.return_value = negated_evaluation
    evaluation.item.return_value = "evaluation_as_float"
    return evaluation


@pytest.fixture
def negated_evaluation(negated_evaluation_plus_reg_term):
    negated_evaluation = MagicMock(name="negated_evaluation")
    negated_evaluation.__add__.return_value = negated_evaluation_plus_reg_term
    return negated_evaluation


@pytest.fixture
def negated_evaluation_plus_reg_term():
    return MagicMock(name="negated_evaluation + reg_term")


@pytest.fixture
def current_input(initial_input):
    return initial_input


@pytest.fixture
def initial_input(cloned_initial_input):
    initial_input = MagicMock()
    initial_input.clone.return_value = cloned_initial_input
    initial_input.grad = "grad"
    cloned_grad_prop = PropertyMock(side_effect=lambda: "cloned_" + initial_input.grad)
    type(initial_input).cloned_grad = cloned_grad_prop
    initial_input.data = "mei_data"
    cloned_data_prop = PropertyMock(side_effect=lambda: "cloned_" + initial_input.data)
    type(initial_input).cloned_data = cloned_data_prop
    initial_input.extract.return_value = "current_input"
    return initial_input


@pytest.fixture
def cloned_initial_input():
    cloned_initial_input = MagicMock(name="cloned_initial_input")
    cloned_initial_input.__repr__ = MagicMock(return_value="initial_input")
    return cloned_initial_input


@pytest.fixture
def optimizer():
    optimizer = MagicMock(name="optimizer")
    optimizer.__repr__ = MagicMock(return_value="optimizer")
    return optimizer


@pytest.fixture
def transform(transformed_mei):
    transform = MagicMock(name="transform", return_value=transformed_mei)
    transform.__repr__ = MagicMock(return_value="transform")
    return transform


@pytest.fixture
def transformed_mei():
    transformed_mei = MagicMock(name="transformed_mei")
    transformed_mei.data.cpu.return_value.clone.return_value = "cloned_transformed_mei_data"
    return transformed_mei


@pytest.fixture
def regularization(reg_term):
    regularization = MagicMock(name="regularization", return_value=reg_term)
    regularization.__repr__ = MagicMock(return_value="regularization")
    return regularization


@pytest.fixture
def reg_term():
    reg_term = MagicMock(name="reg_term", spec=Tensor)
    reg_term.item.return_value = "reg_term_as_float"
    return reg_term


@pytest.fixture
def precondition():
    precondition = MagicMock(name="precondition", return_value="preconditioned_grad")
    precondition.__repr__ = MagicMock(return_value="precondition")
    return precondition


@pytest.fixture
def postprocessing():
    postprocessing = MagicMock(name="postprocessing", return_value="post_processed_mei_data")
    postprocessing.__repr__ = MagicMock(return_value="postprocessing")
    return postprocessing


class TestInit:
    def test_if_func_gets_stored_as_instance_attribute(self, mei, func):
        assert mei().func is func

    def test_if_input_cls_gets_correctly_initialized(self, mei, input_cls, initial_input_tensor):
        mei()
        input_cls.assert_called_once_with(initial_input_tensor)

    def test_if_clone_of_initial_input_gets_stored_as_instance_attribute(self, mei, cloned_initial_input):
        assert mei().initial is cloned_initial_input

    def test_if_optimizer_gets_stored_as_instance_attribute(self, mei, optimizer):
        assert mei().optimizer is optimizer

    def test_if_transform_gets_stored_as_instance_attribute_if_provided(self, mei, transform):
        assert mei().transform is transform

    def test_if_transform_is_default_transform_if_not_provided(self, func, initial_input, optimizer):
        assert optimization.MEI(func, initial_input, optimizer).transform is optimization.default_transform

    def test_if_regularization_gets_stored_as_instance_attribute_if_provided(self, mei, regularization):
        assert mei().regularization is regularization

    def test_if_regularization_is_default_regularization_if_not_provided(self, func, initial_input, optimizer):
        assert optimization.MEI(func, initial_input, optimizer).regularization is optimization.default_regularization

    def test_if_precondition_gets_stored_as_instance_attribute_if_provided(self, mei, precondition):
        assert mei().precondition is precondition

    def test_if_precondition_is_default_precondition_if_not_provided(self, func, initial_input, optimizer):
        assert optimization.MEI(func, initial_input, optimizer).precondition is optimization.default_precondition


class TestEvaluate:
    @pytest.mark.parametrize("n_steps", [0, 1, 10])
    def test_if_transform_is_correctly_called(self, mei, transform, current_input, n_steps):
        mei = mei()
        for _ in range(n_steps):
            mei.step()
        mei.evaluate()
        calls = [call(current_input.tensor, i) for i in range(n_steps)] + [call(current_input.tensor, n_steps)]
        transform.assert_has_calls(calls)

    def test_if_func_is_correctly_called(self, mei, func, transformed_mei):
        mei().evaluate()
        func.assert_called_once_with(transformed_mei)

    def test_if_evaluate_returns_correct_value(self, mei, evaluation):
        assert mei().evaluate() == evaluation


class TestStep:
    def test_if_optimizer_gradient_is_zeroed(self, mei, optimizer):
        mei().step()
        optimizer.zero_grad.assert_called_with()

    @pytest.mark.parametrize("n_steps", [1, 10])
    def test_if_transform_is_correctly_called(self, mei, transform, current_input, n_steps):
        mei = mei()
        for _ in range(n_steps):
            mei.step()
        calls = [call(current_input.tensor, i) for i in range(n_steps)]
        transform.assert_has_calls(calls)

    def test_if_func_is_correctly_called(self, mei, func, transformed_mei):
        mei().step()
        func.assert_called_once_with(transformed_mei)

    @pytest.mark.parametrize("n_steps", [1, 10])
    def test_if_regularization_is_correctly_called(self, mei, regularization, transformed_mei, n_steps):
        mei = mei()
        for _ in range(n_steps):
            mei.step()
        calls = [call(transformed_mei, i) for i in range(n_steps)]
        assert regularization.mock_calls == calls

    def test_if_evaluation_is_negated(self, mei, evaluation):
        mei().step()
        evaluation.__neg__.assert_called_once_with()

    def test_if_regularization_term_is_added_to_negated_evaluation(self, mei, negated_evaluation, reg_term):
        mei().step()
        negated_evaluation.__add__.assert_called_once_with(reg_term)

    def test_if_backward_is_called_on_negated_evaluation_plus_reg_term(self, mei, negated_evaluation_plus_reg_term):
        mei().step()
        negated_evaluation_plus_reg_term.backward.assert_called_once_with()

    def test_if_runtime_error_is_raised_if_gradient_does_not_reach_mei(self, mei, current_input):
        current_input.grad = None
        with pytest.raises(RuntimeError):
            mei().step()

    @pytest.mark.parametrize("n_steps", [1, 10])
    def test_if_precondition_is_called_correctly(self, mei, precondition, n_steps):
        mei = mei()
        for _ in range(n_steps):
            mei.step()
        calls = [call("grad", 0)] + [call("preconditioned_grad", i) for i in range(1, n_steps)]
        assert precondition.mock_calls == calls

    def test_if_preconditioned_gradient_is_reassigned_to_current_input(self, mei, current_input):
        mei().step()
        assert current_input.grad == "preconditioned_grad"

    def test_if_optimizer_takes_a_step(self, mei, optimizer):
        mei().step()
        optimizer.step.assert_called_once_with()

    @pytest.mark.parametrize("n_steps", [1, 10])
    def test_if_postprocessing_is_called_correctly(self, mei, postprocessing, n_steps):
        mei = mei()
        for _ in range(n_steps):
            mei.step()
        calls = [call("mei_data", 0)] + [call("post_processed_mei_data", i) for i in range(1, n_steps)]
        assert postprocessing.mock_calls == calls

    def test_if_transformed_input_is_transferred_to_cpu_when_added_to_state(self, mei, transformed_mei):
        mei().step()
        transformed_mei.data.cpu.assert_called_once_with()

    def test_if_transformed_input_is_cloned_when_added_to_state(self, mei, transformed_mei):
        mei().step()
        transformed_mei.data.cpu.return_value.clone.assert_called_once_with()

    def test_if_state_cls_is_correctly_called(self, mei, state_cls):
        mei().step()
        state = dict(
            i_iter=0,
            evaluation="evaluation_as_float",
            reg_term="reg_term_as_float",
            grad="cloned_grad",
            preconditioned_grad="cloned_preconditioned_grad",
            input_="cloned_mei_data",
            transformed_input="cloned_transformed_mei_data",
            post_processed_input="cloned_post_processed_mei_data",
        )
        state_cls.from_dict.assert_called_once_with(state)

    def test_if_step_returns_the_correct_value(self, mei):
        assert mei().step() == "state_instance"


def test_repr(mei):
    assert repr(mei()) == (
        (
            "MEI(func, initial_input, optimizer, transform=transform, regularization=regularization, "
            "precondition=precondition, postprocessing=postprocessing)"
        )
    )
