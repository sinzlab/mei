from functools import partial

import pytest
import torch
from torch.optim import SGD

from featurevis import optimization
from featurevis.stoppers import NumIterations
from featurevis.domain import Input
from featurevis.tracking import Tracker


@pytest.fixture
def optimize(mei, stopper):
    return partial(optimization.optimize, mei, stopper, Tracker())


@pytest.fixture
def mei(model, initial_mei, optimizer):
    return optimization.MEI(model, initial_mei, optimizer)


@pytest.fixture
def optimizer(initial_mei):
    return SGD([initial_mei.tensor], lr=0.1)


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
    return Input(torch.zeros(2, 2))


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
    return partial(optimization.optimize, mei_with_transform, stopper, Tracker())


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


@pytest.fixture
def optimize_with_regularization(mei_with_regularization, stopper):
    return partial(optimization.optimize, mei_with_regularization, stopper, Tracker())


@pytest.fixture
def mei_with_regularization(model, initial_mei, optimizer, regularization):
    return optimization.MEI(model, initial_mei, optimizer, regularization=regularization)


@pytest.fixture
def regularization():
    def _regularization(mei, _i_iteration):
        return torch.sum(torch.abs(mei))

    return _regularization


def test_if_optimization_process_converges_when_regularization_is_used(optimize_with_regularization, true_mei):
    _, optimized_mei = optimize_with_regularization()
    assert torch.allclose(optimized_mei, true_mei / 2)


def test_if_final_evaluation_matches_expected_value_when_regularization_is_used(optimize_with_regularization):
    final_evaluation, _ = optimize_with_regularization()
    assert final_evaluation == pytest.approx(-1.0)


@pytest.fixture
def optimize_with_precondition(mei_with_precondition, stopper):
    return partial(optimization.optimize, mei_with_precondition, stopper, Tracker())


@pytest.fixture
def mei_with_precondition(model, initial_mei, optimizer, precondition):
    return optimization.MEI(model, initial_mei, optimizer, precondition=precondition)


@pytest.fixture
def precondition():
    def _precondition(grad, _i_iteration):
        grad[0, 0] = 0.0
        return grad

    return _precondition


def test_if_optimization_process_converges_when_precondition_is_used(optimize_with_precondition, true_mei):
    _, optimized_mei = optimize_with_precondition()
    true_mei[0, 0] = 0.0
    assert torch.allclose(optimized_mei, true_mei)


def test_if_final_evaluation_matches_expected_value_when_precondition_is_used(optimize_with_precondition):
    final_evaluation, _ = optimize_with_precondition()
    assert final_evaluation == pytest.approx(-1.0)


@pytest.fixture
def optimize_with_postprocessing(mei_with_postprocessing, stopper):
    return partial(optimization.optimize, mei_with_postprocessing, stopper, Tracker())


@pytest.fixture
def mei_with_postprocessing(model, initial_mei, optimizer, postprocessing):
    return optimization.MEI(model, initial_mei, optimizer, postprocessing=postprocessing)


@pytest.fixture
def postprocessing():
    def _postprocessing(mei_data, _i_iteration):
        return mei_data / mei_data.abs().sum()

    return _postprocessing


def test_if_optimization_process_converges_when_postprocessing_is_used(optimize_with_postprocessing, true_mei):
    _, optimized_mei = optimize_with_postprocessing()
    assert torch.allclose(optimized_mei, true_mei / 4)
