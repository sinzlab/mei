from unittest.mock import MagicMock, call
from functools import partial

import pytest
from torch import Tensor

from mei import optimization
from mei.stoppers import OptimizationStopper
from mei.domain import State
from mei.tracking import Tracker


class TestDefaults:
    @pytest.fixture
    def mei(self):
        return MagicMock(name="mei")

    def test_default_transform(self, mei):
        assert optimization.default_transform(mei, 0) == mei

    def test_default_regularization(self, mei):
        reg_term = optimization.default_regularization(mei, 0)
        assert isinstance(reg_term, Tensor) and reg_term == 0.0

    def test_default_precondition(self):
        grad = MagicMock(name="grad")
        assert optimization.default_precondition(grad, 0) == grad

    def test_default_postprocessing(self, mei):
        assert optimization.default_postprocessing(mei, 0) == mei


class TestOptimize:
    @pytest.fixture
    def optimize(self, mei, tracker):
        return partial(optimization.optimize, mei, tracker=tracker)

    @pytest.fixture
    def mei(self, current_state):
        mei = MagicMock(name="mei", return_value="mei", spec=optimization.MEI)
        mei.step.return_value = current_state
        mei.current_input = "current_input"
        return mei

    @pytest.fixture
    def current_state(self):
        current_state = MagicMock(
            name="current_state", spec=State, evaluation="evaluation", post_processed_input="post_processed_input"
        )
        return current_state

    @pytest.fixture
    def stopper(self):
        def _stopper(num_iterations=1):
            return MagicMock(
                name="stopper",
                side_effect=[(False, "stopper_output") for _ in range(num_iterations)] + [(True, "stopper_output")],
                spec=OptimizationStopper,
            )

        return _stopper

    @pytest.fixture
    def tracker(self):
        return MagicMock(name="tracker", spec=Tracker)

    @pytest.fixture(params=[0, 1, 100])
    def num_iterations(self, request):
        return request.param

    def test_if_mei_takes_steps_correctly(self, optimize, mei, stopper, num_iterations):
        optimize(stopper(num_iterations))
        calls = [call() for _ in range(num_iterations + 1)]
        assert mei.step.mock_calls == calls

    def test_if_stopper_is_called_correctly(self, optimize, stopper, current_state, num_iterations):
        stopper = stopper(num_iterations)
        optimize(stopper)
        calls = [call(current_state) for _ in range(num_iterations + 1)]
        assert stopper.mock_calls == calls

    def test_if_stopper_output_is_assigned_to_current_state(self, optimize, stopper, current_state):
        optimize(stopper())
        assert current_state.stopper_output == "stopper_output"

    def test_if_tracker_is_called_correctly(self, optimize, current_state, stopper, tracker, num_iterations):
        optimize(stopper(num_iterations))
        calls = [call(current_state) for _ in range(num_iterations + 1)]
        assert tracker.track.mock_calls == calls

    def test_if_result_is_correctly_returned(self, optimize, stopper):
        result = optimize(stopper())
        assert result == ("evaluation", "post_processed_input")
