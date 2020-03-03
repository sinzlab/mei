from unittest.mock import MagicMock
from functools import partial

import pytest

from featurevis import methods


class TestGradientAscent:
    @pytest.fixture
    def gradient_ascent(self, dataloaders, config, import_object, get_dims, get_initial_guess, ascend):
        return partial(
            methods.gradient_ascent,
            dataloaders,
            "model",
            config,
            import_object=import_object,
            get_dims=get_dims,
            get_initial_guess=get_initial_guess,
            ascend=ascend,
        )

    @pytest.fixture
    def dataloaders(self):
        return dict(train=dict(session1=None))

    @pytest.fixture
    def config(self):
        return dict(
            optim_kwargs=None, transform=None, regularization="module.function", gradient_f=None, post_update=None
        )

    @pytest.fixture
    def import_object(self):
        return MagicMock(return_value="imported_function")

    @pytest.fixture
    def get_dims(self):
        return MagicMock(return_value=dict(session1=dict(inputs=(100, 10, 24, 24))))

    @pytest.fixture
    def get_initial_guess(self):
        return MagicMock(return_value="initial_guess")

    @pytest.fixture
    def ascend(self):
        return MagicMock(return_value=("mei", "evaluations", "_"))

    def test_if_import_object_is_correctly_called(self, gradient_ascent, import_object):
        gradient_ascent()
        import_object.assert_called_once_with("module.function")

    def test_if_get_dims_is_correctly_called(self, gradient_ascent, get_dims):
        gradient_ascent()
        get_dims.assert_called_once_with(dict(session1=None))

    def test_if_get_initial_guess_is_correctly_called(self, gradient_ascent, get_initial_guess):
        gradient_ascent()
        get_initial_guess.assert_called_once_with(1, 10, 24, 24)

    def test_if_ascend_is_correctly_called(self, gradient_ascent, ascend):
        gradient_ascent()
        ascend.assert_called_once_with(
            "model",
            "initial_guess",
            optim_kwargs=dict(),
            transform=None,
            regularization="imported_function",
            gradient_f=None,
            post_update=None,
        )

    def test_if_mei_and_evaluations_are_returned(self, gradient_ascent):
        assert gradient_ascent() == ("mei", "evaluations")
