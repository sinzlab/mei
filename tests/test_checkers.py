from unittest.mock import MagicMock

import pytest

from featurevis import checkers


class TestNumIterations:
    @pytest.fixture
    def optimized(self):
        def _optimized(n):
            return checkers.NumIterations(n)

        return _optimized

    def test_init(self, optimized):
        assert optimized(5).num_iterations == 5

    @pytest.mark.parametrize("num_iterations", [0, 1, 1000])
    def test_call(self, optimized, num_iterations):
        optimized = optimized(num_iterations)
        mei = MagicMock(name="mei")
        evaluation = 0.5
        for _ in range(num_iterations):
            assert optimized(mei, evaluation) is False
        assert optimized(mei, evaluation) is True

    def test_repr(self, optimized):
        assert optimized(5).__repr__() == f"NumIterations(5)"
