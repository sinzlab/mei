from abc import ABC
from typing import Any
from unittest.mock import MagicMock, call
from contextlib import contextmanager

import pytest

from mei import objectives
from mei.domain import State


@contextmanager
def does_not_raise():
    yield


class TestObjective:
    @pytest.fixture
    def fake_objective(self):
        class FakeObjective(objectives.Objective):
            compute = MagicMock(name="FakeObjective().compute", return_value="result")

        return FakeObjective

    def test_if_objective_is_subclass_of_abc(self):
        assert issubclass(objectives.Objective, ABC)

    def test_if_compute_is_called_correctly(self, fake_objective):
        fake_objective()("current_state")
        fake_objective.compute.assert_called_once_with("current_state")

    def test_if_computed_result_is_returned(self, fake_objective):
        assert fake_objective()("current_state") == "result"


class TestRegularIntervalObjective:
    @pytest.fixture
    def fake_objective(self):
        class FakeObjective(objectives.RegularIntervalObjective):
            __qualname__ = "FakeObjective"

            def __init__(self, interval):
                super().__init__(interval)
                self.compute_mock = MagicMock(name="compute_mock", spec=self.compute)

            def compute(self, current_state: State) -> Any:
                return self.compute_mock(current_state)

        return FakeObjective

    def test_if_regular_interval_objective_is_subclass_of_objective(self):
        assert issubclass(objectives.RegularIntervalObjective, objectives.Objective)

    def test_init(self, fake_objective):
        assert fake_objective(10).interval == 10

    @pytest.mark.parametrize(
        "interval,expectation", [(-1, pytest.raises(ValueError)), (0, pytest.raises(ValueError)), (1, does_not_raise())]
    )
    def test_if_interval_equal_to_or_smaller_than_zero_raises_value_error(self, fake_objective, interval, expectation):
        with expectation:
            fake_objective(interval)

    @pytest.mark.parametrize("n_states", [1, 10])
    @pytest.mark.parametrize("interval", [1, 2, 10, 11])
    def test_if_compute_method_is_called_correctly(self, fake_objective, n_states, interval):
        states = [MagicMock(name="state" + str(i), spec=State, i_iter=i) for i in range(n_states)]
        obj = fake_objective(interval)
        for current_state in states:
            obj(current_state)
        calls = [call(s) for s in states if s.i_iter % interval == 0]
        assert obj.compute_mock.mock_calls == calls

    def test_repr(self, fake_objective):
        assert repr(fake_objective(2)) == "FakeObjective(2)"


class TestEvaluationObjective:
    def test_if_objective_is_subclass_of_regular_interval_objective(self):
        assert issubclass(objectives.EvaluationObjective, objectives.RegularIntervalObjective)

    def test_if_correct_value_is_computed(self):
        obj = objectives.EvaluationObjective(1)
        current_state = MagicMock(name="current_state", spec=State)
        current_state.evaluation = "evaluation"
        assert obj.compute(current_state) == "evaluation"


class TestPostProcessedInputObjective:
    def test_if_objective_is_subclass_of_regular_interval_objective(self):
        assert issubclass(objectives.PostProcessedInputObjective, objectives.RegularIntervalObjective)

    def test_if_correct_value_is_computed(self):
        obj = objectives.PostProcessedInputObjective(1)
        current_state = MagicMock(name="current_state", spec=State, post_processed_input="post_processed_input")
        assert obj.compute(current_state) == "post_processed_input"
