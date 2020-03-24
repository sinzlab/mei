from functools import partial

import pytest
import torch
from torch.optim import SGD

from featurevis import optimization
from featurevis.stoppers import NumIterations


@pytest.fixture
def optimize(model, true_mei):
    initial_mei = torch.zeros(2, 2)
    mei = optimization.MEI(model, initial_mei)
    optimizer = SGD([initial_mei], lr=0.1)
    stopper = NumIterations(100)
    return partial(optimization.optimize, mei, optimizer, stopper)


@pytest.fixture
def model(true_mei):
    def _model(current_mei):
        return -(true_mei - current_mei).pow(2).sum()

    return _model


@pytest.fixture
def true_mei():
    return torch.tensor([[-1.0, 1.0], [1.0, -1.0]])


def test_if_optimization_process_converges_to_true_mei(optimize, true_mei):
    _, optimized_mei = optimize()
    assert torch.allclose(optimized_mei, true_mei)


def test_if_final_evaluation_matches_expected_value(optimize):
    final_evaluation, _ = optimize()
    assert final_evaluation == pytest.approx(0.0)
