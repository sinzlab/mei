from unittest.mock import MagicMock
from functools import partial

import pytest

from featurevis import methods


class TestGradientAscent:
    @pytest.fixture
    def gradient_ascent(self, dataloaders, model, config, set_seed, import_object, get_dims, get_initial_guess, ascend):
        return partial(
            methods.gradient_ascent,
            dataloaders,
            model,
            config,
            "seed",
            set_seed=set_seed,
            import_object=import_object,
            get_dims=get_dims,
            get_initial_guess=get_initial_guess,
            ascend=ascend,
        )

    @pytest.fixture
    def dataloaders(self):
        return dict(train=dict(session1=None))

    @pytest.fixture
    def model(self):
        return MagicMock()

    @pytest.fixture
    def config(self):
        return dict(
            optim_kwargs=None,
            transform=None,
            regularization="module.function",
            gradient_f=None,
            post_update=None,
            device="device",
        )

    @pytest.fixture
    def set_seed(self):
        return MagicMock()

    @pytest.fixture
    def import_object(self):
        return MagicMock(return_value="imported_function")

    @pytest.fixture
    def get_dims(self):
        return MagicMock(return_value=dict(session1=dict(inputs=(100, 10, 24, 24))))

    @pytest.fixture
    def get_initial_guess(self, initial_guess):
        return MagicMock(return_value=initial_guess)

    @pytest.fixture
    def initial_guess(self):
        return MagicMock()

    @pytest.fixture
    def ascend(self, mei):
        return MagicMock(return_value=(mei, [1, 2, 3, 4, 5], []))

    @pytest.fixture
    def mei(self):
        mei = MagicMock()
        mei.cpu.return_value = "mei"
        return mei

    def test_if_import_object_is_correctly_called(self, gradient_ascent, import_object):
        gradient_ascent()
        import_object.assert_called_once_with("module.function")

    def test_if_get_dims_is_correctly_called(self, gradient_ascent, get_dims):
        gradient_ascent()
        get_dims.assert_called_once_with(dict(session1=None))

    def test_if_get_initial_guess_is_correctly_called(self, gradient_ascent, get_initial_guess):
        gradient_ascent()
        get_initial_guess.assert_called_once_with(1, 10, 24, 24, device="device")

    def test_if_ascend_is_correctly_called(self, gradient_ascent, ascend, model, initial_guess):
        gradient_ascent()
        ascend.assert_called_once_with(
            model,
            initial_guess,
            optim_kwargs=dict(),
            transform=None,
            regularization="imported_function",
            gradient_f=None,
            post_update=None,
        )

    def test_if_mei_and_evaluations_are_returned(self, gradient_ascent):
        assert gradient_ascent() == ("mei", 5, dict(function_evaluations=[1, 2, 3, 4, 5], regularization_terms=[]))

    def test_if_seed_is_correctly_set(self, gradient_ascent, set_seed):
        gradient_ascent()
        set_seed.assert_called_once_with("seed")

    def test_if_eval_method_of_model_is_called(self, gradient_ascent, model):
        gradient_ascent()
        model.eval.assert_called_once_with()

    def test_if_model_is_transferred_to_device(self, gradient_ascent, model):
        gradient_ascent()
        model.to.assert_called_once_with("device")
