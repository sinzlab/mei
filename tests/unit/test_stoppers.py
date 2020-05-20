from unittest.mock import MagicMock

import pytest

from featurevis import stoppers


class TestNumIterations:
    @pytest.fixture
    def stopper(self):
        def _stopper(n):
            return stoppers.NumIterations(n)

        return _stopper

    def test_init(self, stopper):
        assert stopper(5).num_iterations == 5

    @pytest.mark.parametrize("num_iterations", [0, 1, 1000])
    def test_call(self, stopper, num_iterations):
        stopper = stopper(num_iterations)
        current_state = MagicMock(name="current_state")
        for _ in range(num_iterations):
            assert stopper(current_state) == (False, None)
        assert stopper(current_state) == (True, None)

    def test_repr(self, stopper):
        assert stopper(5).__repr__() == f"NumIterations(5)"
