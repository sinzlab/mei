from unittest.mock import MagicMock, call
from functools import partial

import pytest

from featurevis import optimization


class TestMEI:
    @pytest.fixture
    def mei(self, func, initial_guess):
        return partial(optimization.MEI, func, initial_guess)

    @pytest.fixture
    def func(self):
        func = MagicMock(return_value="evaluation")
        representation = MagicMock(return_value="func")
        func.__repr__ = representation
        return func

    @pytest.fixture
    def initial_guess(self):
        initial_guess = MagicMock()
        initial_guess.__repr__ = MagicMock(return_value="initial_guess")
        initial_guess.detach.return_value.squeeze.return_value.cpu.return_value = "final_mei"
        return initial_guess

    @pytest.fixture
    def transform(self, transformed_mei):
        return MagicMock(name="transform", return_value=transformed_mei)

    @pytest.fixture
    def transformed_mei(self):
        return MagicMock(name="transformed_mei")

    def test_if_func_gets_stored_as_instance_attribute(self, mei, func):
        assert mei().func is func

    def test_if_initial_guess_gets_stored_as_instance_attribute(self, mei, initial_guess):
        assert mei().initial_guess is initial_guess

    def test_if_transform_gets_stored_as_instance_attribute_if_provided(self, mei, transform):
        assert mei(transform=transform).transform is transform

    def test_if_transform_is_identity_function_if_not_provided(self, mei):
        assert mei().transform("mei") == "mei"

    def test_if_initial_guess_gets_grad_enabled(self, mei, initial_guess):
        mei()
        initial_guess.requires_grad_.assert_called_once_with()

    def test_if_transform_is_correctly_called(self, mei, transform, initial_guess):
        mei(transform=transform).evaluate(0)
        transform.assert_called_once_with(initial_guess, i_iteration=0)

    def test_if_func_is_correctly_called(self, mei, func, transform, transformed_mei):
        mei(transform=transform).evaluate(0)
        func.assert_called_once_with(transformed_mei)

    def test_if_evaluate_returns_the_correct_value(self, mei):
        assert mei().evaluate(0) == "evaluation"

    def test_if_mei_is_detached_when_retrieved(self, mei, initial_guess):
        mei().get_mei()
        initial_guess.detach.assert_called_once_with()

    def test_if_cloned_mei_is_squeezed_when_retrieved(self, mei, initial_guess):
        mei().get_mei()
        initial_guess.detach.return_value.squeeze.assert_called_once_with()

    def test_if_squeezed_mei_is_switched_to_cpu_when_retrieved(self, mei, initial_guess):
        mei().get_mei()
        initial_guess.detach.return_value.squeeze.return_value.cpu.assert_called_once_with()

    def test_if_cloned_mei_is_returned_when_retrieved(self, mei, initial_guess):
        assert mei().get_mei() == "final_mei"

    def test_repr(self, mei):
        assert mei().__repr__() == "MEI(func, initial_guess)"


class TestOptimize:
    @pytest.fixture
    def optimize(self, mei, optimizer):
        return partial(optimization.optimize, mei, optimizer)

    @pytest.fixture
    def mei(self, evaluation):
        mei = MagicMock(return_value="mei")
        mei.evaluate.return_value = evaluation
        mei.get_mei.return_value = "mei"
        return mei

    @pytest.fixture
    def optimizer(self):
        return MagicMock()

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

    def test_if_optimized_is_called_correctly(self, optimize, mei, optimized, evaluation):
        optimized = optimized()
        optimize(optimized)
        optimized.assert_called_with(mei, evaluation)

    def test_if_optimizer_gradient_gets_zeroed_correctly(self, optimize, optimizer, optimized):
        optimize(optimized())
        optimizer.zero_grad.assert_called_with()

    def test_if_mei_is_evaluated_correctly(self, optimize, mei, optimized, num_iterations):
        optimize(optimized(num_iterations))
        calls = [call(0)] + [call(i) for i in range(num_iterations)]
        mei.evaluate.assert_has_calls(calls)
        assert mei.evaluate.call_count == len(calls)

    def test_if_evaluation_is_negated(self, optimize, optimized, evaluation):
        optimize(optimized())
        evaluation.__neg__.assert_called()

    def test_if_backward_is_called_correctly_on_negated_evaluation(self, optimize, optimized, negated_evaluation):
        optimize(optimized())
        negated_evaluation.backward.assert_called_with()

    def test_if_optimizer_takes_step_correctly(self, optimize, optimizer, optimized):
        optimize(optimized())
        optimizer.step.assert_called_with()

    def test_if_optimized_is_called_correct_number_of_times(self, optimize, optimized, num_iterations):
        optimized = optimized(num_iterations)
        optimize(optimized)
        assert optimized.call_count == num_iterations + 1

    def test_if_optimizer_gradient_is_zeroed_correct_number_of_times(
        self, optimize, optimizer, optimized, num_iterations
    ):
        optimize(optimized(num_iterations))
        assert optimizer.zero_grad.call_count == num_iterations

    def test_if_optimizer_takes_correct_number_of_steps(self, optimize, optimizer, optimized, num_iterations):
        optimize(optimized(num_iterations))
        assert optimizer.step.call_count == num_iterations

    def test_if_mei_is_correctly_called(self, mei, optimize, optimized):
        optimize(optimized())
        mei.get_mei.assert_called_once_with()

    def test_if_result_is_correctly_returned(self, optimize, optimized):
        result = optimize(optimized())
        assert result == ("evaluation", "mei")
