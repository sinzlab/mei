from unittest.mock import MagicMock, call
from functools import partial
import dataclasses

import pytest

from featurevis import optimization


class TestGradient:
    @pytest.fixture
    def regular(self):
        return MagicMock(name="regular")

    @pytest.fixture
    def preconditioned(self):
        return MagicMock(name="preconditioned")

    @pytest.fixture
    def fields(self, regular, preconditioned):
        gradient = optimization.Gradient(regular, preconditioned)
        fields = dataclasses.fields(gradient)
        return fields

    def test_if_dataclass(self):
        assert dataclasses.dataclass(optimization.Gradient)

    def test_field_names(self, fields):
        assert all(n == f.name for n, f in zip(("regular", "preconditioned"), fields))

    def test_field_types(self, fields):
        assert all(f.type == "Tensor" for f in fields)


class TestDefaults:
    @pytest.fixture
    def mei(self):
        return MagicMock(name="mei")

    def test_default_transform(self, mei):
        assert optimization.default_transform(mei, 0) == mei

    def test_default_regularization(self, mei):
        assert optimization.default_regularization(mei, 0) == 0

    def test_default_precondition(self):
        gradient = MagicMock(name="gradient")
        assert optimization.default_precondition(gradient, 0) == gradient

    def test_default_postprocessing(self, mei):
        assert optimization.default_postprocessing(mei, 0) == mei


class TestMEI:
    @pytest.fixture
    def mei(self, func, initial, optimizer, transform, regularization, precondition, postprocessing):
        return partial(
            optimization.MEI,
            func,
            initial,
            optimizer,
            transform=transform,
            regularization=regularization,
            precondition=precondition,
            postprocessing=postprocessing,
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
    def current_input(self, initial):
        return initial

    @pytest.fixture
    def initial(self, cloned_initial):
        initial = MagicMock()
        initial.clone.return_value = cloned_initial
        initial.gradient = "gradient"
        initial.data = "mei_data"
        initial.extract.return_value = "current_input"
        return initial

    @pytest.fixture
    def cloned_initial(self):
        cloned_initial = MagicMock(name="cloned_initial")
        cloned_initial.__repr__ = MagicMock(return_value="initial")
        return cloned_initial

    @pytest.fixture
    def optimizer(self):
        optimizer = MagicMock(name="optimizer")
        optimizer.__repr__ = MagicMock(return_value="optimizer")
        return optimizer

    @pytest.fixture
    def transform(self, transformed_mei):
        transform = MagicMock(name="transform", return_value=transformed_mei)
        transform.__repr__ = MagicMock(return_value="transform")
        return transform

    @pytest.fixture
    def transformed_mei(self):
        return MagicMock(name="transformed_mei")

    @pytest.fixture
    def regularization(self):
        regularization = MagicMock(name="regularization", return_value="reg_term")
        regularization.__repr__ = MagicMock(return_value="regularization")
        return regularization

    @pytest.fixture
    def precondition(self):
        precondition = MagicMock(name="precondition", return_value="preconditioned_gradient")
        precondition.__repr__ = MagicMock(return_value="precondition")
        return precondition

    @pytest.fixture
    def postprocessing(self):
        postprocessing = MagicMock(name="postprocessing", return_value="post_processed_mei_data")
        postprocessing.__repr__ = MagicMock(return_value="postprocessing")
        return postprocessing

    class TestInit:
        def test_if_func_gets_stored_as_instance_attribute(self, mei, func):
            assert mei().func is func

        def test_if_clone_of_initial_guess_gets_stored_as_instance_attribute(self, mei, cloned_initial):
            assert mei().initial is cloned_initial

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

        def test_if_regularization_term_is_added_to_negated_evaluation(self, mei, negated_evaluation):
            mei().step()
            negated_evaluation.__add__.assert_called_once_with("reg_term")

        def test_if_backward_is_called_on_negated_evaluation_plus_reg_term(self, mei, negated_evaluation_plus_reg_term):
            mei().step()
            negated_evaluation_plus_reg_term.backward.assert_called_once_with()

        def test_if_runtime_error_is_raised_if_gradient_does_not_reach_mei(self, mei, current_input):
            current_input.gradient = None
            with pytest.raises(RuntimeError):
                mei().step()

        @pytest.mark.parametrize("n_steps", [1, 10])
        def test_if_precondition_is_called_correctly(self, mei, precondition, n_steps):
            mei = mei()
            for _ in range(n_steps):
                mei.step()
            calls = [call("gradient", 0)] + [call("preconditioned_gradient", i) for i in range(1, n_steps)]
            assert precondition.mock_calls == calls

        def test_if_preconditioned_gradient_is_reassigned_to_current_input(self, mei, current_input):
            mei().step()
            assert current_input.gradient == "preconditioned_gradient"

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

        def test_if_step_returns_the_correct_value(self, mei, evaluation):
            assert mei().step() == evaluation

    def test_if_extract_is_correctly_called_when_accessing_current_input(self, mei, current_input):
        _ = mei().current_input
        current_input.extract.assert_called_once_with()

    def test_return_value_of_current_input_property(self, mei):
        assert mei().current_input == "current_input"

    def test_repr(self, mei):
        assert repr(mei()) == (
            (
                "MEI(func, initial, optimizer, transform=transform, regularization=regularization, "
                "precondition=precondition, postprocessing=postprocessing)"
            )
        )


class TestOptimize:
    @pytest.fixture
    def optimize(self, mei):
        return partial(optimization.optimize, mei)

    @pytest.fixture
    def mei(self, evaluation):
        mei = MagicMock(return_value="mei")
        mei.step.return_value = evaluation
        mei.current_input = "current_input"
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
        assert result == ("evaluation", "current_input")
