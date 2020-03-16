import pytest
import torch

from featurevis import core


@pytest.fixture
def func(optimal_x):
    def _func(x):
        return -torch.pow(torch.sum(torch.abs(optimal_x - x)), 2)

    return _func


@pytest.fixture
def optimal_x():
    return torch.tensor([[1.0, -1.0], [-1.0, 1.0]])


@pytest.fixture
def initial_x():
    return torch.zeros(2, 2)


def test_if_estimated_x_matches_optimal_x(func, initial_x, optimal_x):
    estimated_x, *_ = core.gradient_ascent(func, initial_x)
    assert torch.allclose(estimated_x, optimal_x)
