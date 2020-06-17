from unittest.mock import MagicMock, call

import pytest

from mei import tracking
from mei.domain import State


class TestTracker:
    @pytest.fixture
    def tracker(self, objectives):
        return tracking.Tracker(**objectives)

    @pytest.fixture
    def objectives(self, n_objectives, interval):
        objectives = dict()
        for i in range(n_objectives):

            def obj_side_effect(current_state, i_obj=i):
                if current_state.i_iter % interval == 0:
                    return f"obj{i_obj}_result{current_state.i_iter}"

            objectives["obj" + str(i)] = MagicMock(name="obj" + str(i), side_effect=obj_side_effect)
        return objectives

    @pytest.fixture(params=[0, 1, 10])
    def n_objectives(self, request):
        return request.param

    @pytest.fixture(params=[1, 10, 11])
    def interval(self, request):
        return request.param

    @pytest.fixture
    def states(self, n_states):
        return tuple(MagicMock(name="current_state", i_iter=i, spec=State) for i in range(n_states))

    @pytest.fixture(params=[0, 1, 10])
    def n_states(self, request):
        return request.param

    @pytest.fixture
    def expected_log(self, n_objectives, n_states, interval):
        return {
            f"obj{o}": dict(
                times=[s for s in range(n_states) if s % interval == 0],
                values=[f"obj{o}_result{s}" for s in range(n_states) if s % interval == 0],
            )
            for o in range(n_objectives)
        }

    def test_init(self, tracker, objectives):
        assert tracker.objectives == objectives

    def test_if_track_calls_objectives_correctly(self, tracker, objectives, states):
        for current_state in states:
            tracker.track(current_state)
        calls = [call(c_s) for c_s in states]
        assert all(obj.mock_calls == calls for obj in objectives.values())

    def test_if_objectives_are_correctly_represented_in_log(self, tracker, objectives, states, expected_log):
        for current_state in states:
            tracker.track(current_state)
        assert tracker.log == expected_log

    def test_repr(self, tracker, objectives):
        assert tracker.__repr__() == f"Tracker({', '.join(objectives)})"
