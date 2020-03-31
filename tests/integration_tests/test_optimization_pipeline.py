from functools import partial

import pytest
import torch
from torch.optim import SGD

from featurevis import optimization
from featurevis.stoppers import NumIterations


@pytest.fixture
def optimize(mei, stopper):
    return partial(optimization.optimize, mei, stopper)


@pytest.fixture
def mei(model, initial_mei, optimizer):
    return optimization.MEI(model, initial_mei, optimizer)


@pytest.fixture
def optimizer(initial_mei):
    return SGD([initial_mei], lr=0.1)


@pytest.fixture
def stopper():
    return NumIterations(100)


@pytest.fixture
def model(true_mei):
    def _model(current_mei):
        return -(true_mei - current_mei).pow(2).sum()

    return _model


@pytest.fixture
def initial_mei():
    return torch.zeros(2, 2)


@pytest.fixture
def true_mei():
    return torch.tensor([[-1.0, 1.0], [1.0, -1.0]])


def test_if_optimization_process_converges_to_true_mei(optimize, true_mei):
    _, optimized_mei = optimize()
    assert torch.allclose(optimized_mei, true_mei)


def test_if_final_evaluation_matches_expected_value(optimize):
    final_evaluation, _ = optimize()
    assert final_evaluation == pytest.approx(0.0)


@pytest.fixture
def optimize_with_transform(mei_with_transform, stopper):
    return partial(optimization.optimize, mei_with_transform, stopper)


@pytest.fixture
def mei_with_transform(model, initial_mei, optimizer, transform):
    return optimization.MEI(model, initial_mei, optimizer, transform=transform)


@pytest.fixture
def transform():
    def _transform(mei, _i_iteration):
        return -mei

    return _transform


def test_if_optimization_process_converges_to_transformed_mei(optimize_with_transform, transform, true_mei):
    _, optimized_mei = optimize_with_transform()
    assert torch.allclose(optimized_mei, transform(true_mei, 0))
