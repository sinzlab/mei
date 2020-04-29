from abc import ABC
from unittest.mock import MagicMock

from featurevis import objectives
from featurevis.domain import State


def test_if_objective_is_subclass_of_abc():
    assert issubclass(objectives.Objective, ABC)


class TestEvaluation:
    def test_if_evaluation_objective_is_subclass_of_objective(self):
        assert issubclass(objectives.Evaluation, objectives.Objective)

    def test_if_correct_value_is_computed(self):
        obj = objectives.Evaluation()
        current_state = MagicMock(name="current_state", spec=State)
        current_state.evaluation = "evaluation"
        assert obj.compute(current_state) == "evaluation"
