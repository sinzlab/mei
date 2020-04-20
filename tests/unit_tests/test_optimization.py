from unittest.mock import MagicMock, call
from functools import partial

import pytest

from featurevis import optimization


def test_default_transform():
    assert optimization.default_transform("mei", 0) == "mei"


def test_default_regularization():
    assert optimization.default_regularization("mei", 0) == 0


def test_default_precondition():
    assert optimization.default_precondition("gradient", 0) == "gradient"


def test_default_postprocess():
    assert optimization.default_postprocess("mei", 0) == "mei"


class TestMEI:
    @pytest.fixture
    def mei(self, func, initial, optimizer, transform, regularization, precondition):
        return partial(
            optimization.MEI,
            func,
            initial,
            optimizer,
            transform=transform,
            regularization=regularization,
            precondition=precondition,
        )

    @pytest.fixture
    def func(self, evaluation):
        func = MagicMock(return_value=evaluation)
        representation = MagicMock(return_value="func")
        func.__repr__ = representation
        return func

    @pytest.fixture
    def evaluation(self, negated_evaluation):
        evaluation = MagicMock(name="evaluation")
        evaluation.__neg__.return_value = negated_evaluation
        return evaluation

    @pytest.fixture
    def negated_evaluation(self, negated_evaluation_plus_reg_term):
        negated_evaluation = MagicMock(name="negated_evaluation")
        negated_evaluation.__add__.return_value = negated_evaluation_plus_reg_term
        return negated_evaluation

    @pytest.fixture
    def negated_evaluation_plus_reg_term(self):
        return MagicMock(name="negated_evaluation + reg_term")

    @pytest.fixture
    def initial(self):
        initial = MagicMock()
        initial.__repr__ = MagicMock(return_value="initial")
        initial.grad = "gradient"
        initial.detach.return_value.squeeze.return_value.cpu.return_value = "final_mei"
        return initial

    @pytest.fixture
    def optimizer(self):
        return MagicMock(name="optimizer")

    @pytest.fixture
    def transform(self, transformed_mei):
        return MagicMock(name="transform", return_value=transformed_mei)

    @pytest.fixture
    def transformed_mei(self):
        return MagicMock(name="transformed_mei")

    @pytest.fixture
    def regularization(self):
        return MagicMock(name="regularization", return_value="reg_term")

    @pytest.fixture
    def precondition(self):
        return MagicMock(name="precondition", return_value="preconditioned_gradient")

    class TestInit:
        def test_if_func_gets_stored_as_instance_attribute(self, mei, func):
            assert mei().func is func

        def test_if_initial_guess_gets_stored_as_instance_attribute(self, mei, initial):
            assert mei().initial is initial

        def test_if_optimizer_gets_stored_as_instance_attribute(self, mei, optimizer):
            assert mei().optimizer is optimizer

        def test_if_transform_gets_stored_as_instance_attribute_if_provided(self, mei, transform):
            assert mei().transform is transform

        def test_if_transform_is_default_transform_if_not_provided(self, func, initial, optimizer):
            assert optimization.MEI(func, initial, optimizer).transform is optimization.default_transform

        def test_if_regularization_gets_stored_as_instance_attribute_if_provided(self, mei, regularization):
            assert mei().regularization is regularization

        def test_if_regularization_is_default_regularization_if_not_provided(self, func, initial, optimizer):
            assert optimization.MEI(func, initial, optimizer).regularization is optimization.default_regularization

        def test_if_precondition_gets_stored_as_instance_attribute_if_provided(self, mei, precondition):
            assert mei().precondition is precondition

        def test_if_precondition_is_default_precondition_if_not_provided(self, func, initial, optimizer):
            assert optimization.MEI(func, initial, optimizer).precondition is optimization.default_precondition

        def test_if_initial_guess_gets_grad_enabled(self, mei, initial):
            mei()
            initial.requires_grad_.assert_called_once_with()

    class TestEvaluate:
        @pytest.mark.parametrize("n_steps", [0, 1, 10])
        def test_if_transform_is_correctly_called(self, mei, transform, initial, n_steps):
            mei = mei()
            for _ in range(n_steps):
                mei.step()
            mei.evaluate()
            calls = [call(initial, i) for i in range(n_steps)] + [call(initial, n_steps)]
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
        def test_if_transform_is_correctly_called(self, mei, transform, initial, n_steps):
            mei = mei()
            for _ in range(n_steps):
                mei.step()
            calls = [call(initial, i) for i in range(n_steps)]
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

        def test_if_regularization_term_is_added_to_negated_evaluation(self, mei, negated_evaluation):
            mei().step()
            negated_evaluation.__add__.assert_called_once_with("reg_term")

        def test_if_backward_is_called_on_negated_evaluation_plus_reg_term(self, mei, negated_evaluation_plus_reg_term):
            mei().step()
            negated_evaluation_plus_reg_term.backward.assert_called_once_with()

        @pytest.mark.parametrize("n_steps", [1, 10])
        def test_if_precondition_is_called_correctly(self, mei, precondition, n_steps):
            mei = mei()
            for _ in range(n_steps):
                mei.step()
            calls = [call("gradient", 0)] + [call("preconditioned_gradient", i) for i in range(1, n_steps)]
            assert precondition.mock_calls == calls

        def test_if_preconditioned_gradient_is_reassigned_mei(self, mei, initial):
            mei().step()
            assert initial.grad == "preconditioned_gradient"

        def test_if_optimizer_takes_a_step(self, mei, optimizer):
            mei().step()
            optimizer.step.assert_called_once_with()

        def test_if_step_returns_the_correct_value(self, mei, evaluation):
            assert mei().step() == evaluation

    class TestGetMEI:
        def test_if_mei_is_detached_when_retrieved(self, mei, initial):
            _ = mei().mei
            initial.detach.assert_called_once_with()

        def test_if_cloned_mei_is_squeezed_when_retrieved(self, mei, initial):
            _ = mei().mei
            initial.detach.return_value.squeeze.assert_called_once_with()

        def test_if_squeezed_mei_is_switched_to_cpu_when_retrieved(self, mei, initial):
            _ = mei().mei
            initial.detach.return_value.squeeze.return_value.cpu.assert_called_once_with()

        def test_if_cloned_mei_is_returned_when_retrieved(self, mei):
            assert mei().mei == "final_mei"

    def test_repr(self, mei):
        assert mei().__repr__() == "MEI(func, initial)"


class TestOptimize:
    @pytest.fixture
    def optimize(self, mei):
        return partial(optimization.optimize, mei)

    @pytest.fixture
    def mei(self, evaluation):
        mei = MagicMock(return_value="mei")
        mei.step.return_value = evaluation
        mei.mei = "mei"
        return mei

    @pytest.fixture
    def optimized(self):
        def _optimized(num_iterations=1):
            return MagicMock(side_effect=[False for _ in range(num_iterations)] + [True])

        return _optimized

    @pytest.fixture
    def evaluation(self, negated_evaluation):
        evaluation = MagicMock(name="evaluation")
        evaluation.__neg__.return_value = negated_evaluation
        evaluation.item.return_value = "evaluation"
        return evaluation

    @pytest.fixture
    def negated_evaluation(self):
        return MagicMock(name="negated_evaluation")

    @pytest.fixture(params=[0, 1, 100])
    def num_iterations(self, request):
        return request.param

    def test_if_mei_is_evaluated_correctly(self, optimize, mei, optimized):
        optimize(optimized())
        mei.evaluate.assert_called_once_with()

    def test_if_optimized_is_called_correctly(self, optimize, mei, optimized, evaluation):
        optimized = optimized()
        optimize(optimized)
        optimized.assert_called_with(mei, evaluation)

    def test_if_mei_takes_steps_correctly(self, optimize, mei, optimized, num_iterations):
        optimize(optimized(num_iterations))
        calls = [call() for _ in range(num_iterations)]
        mei.step.assert_has_calls(calls)
        assert mei.step.call_count == len(calls)

    def test_if_optimized_is_called_correct_number_of_times(self, optimize, optimized, num_iterations):
        optimized = optimized(num_iterations)
        optimize(optimized)
        assert optimized.call_count == num_iterations + 1

    def test_if_result_is_correctly_returned(self, optimize, optimized):
        result = optimize(optimized())
        assert result == ("evaluation", "mei")
