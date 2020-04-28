from unittest.mock import MagicMock, call

import pytest

from featurevis import tracking
from featurevis.domain import State


class TestTracker:
    @pytest.fixture
    def tracker(self, objectives):
        return tracking.Tracker(**objectives)

    @pytest.fixture(params=[0, 1, 10])
    def objectives(self, request):
        return {
            "obj" + str(i): MagicMock(name=f"obj{i}", side_effect=lambda c_s: f"data{c_s.i_iter}")
            for i in range(request.param)
        }

    @pytest.fixture(params=[0, 1, 10])
    def states(self, request):
        return tuple(MagicMock(name="current_state", i_iter=i, spec=State) for i in range(request.param))

    def test_init(self, tracker, objectives):
        assert tracker.objectives == objectives

    def test_if_track_calls_objectives_correctly(self, tracker, objectives, states):
        for current_state in states:
            tracker.track(current_state)
        calls = [call(c_s) for c_s in states]
        assert all(obj.mock_calls == calls for obj in objectives.values())

    def test_if_objectives_are_correctly_represented_in_log(self, tracker, objectives, states):
        for current_state in states:
            tracker.track(current_state)
        log = {n: ["data" + str(s.i_iter) for s in states] for n in objectives}
        assert tracker.log == log

    def test_repr(self, tracker, objectives):
        assert tracker.__repr__() == f"Tracker({', '.join(objectives)})"
