from unittest.mock import MagicMock
from functools import partial

import pytest

from featurevis import methods


class TestAscend:
    @pytest.fixture
    def config(self):
        return dict(key1="value1", key2="value2")

    @pytest.fixture
    def ascending_func(self, progression, regularization):
        function_evaluations = ["initial_evaluation", "intermittent_evaluation", "final_evaluation"]
        if progression:
            mei = ["initial_mei", "intermittent_mei", "final_mei"]
        else:
            mei = "final_mei"
        if regularization:
            regularization_terms = ["initial_term", "intermittent_term", "final_term"]
        else:
            regularization_terms = []
        return MagicMock(return_value=(mei, function_evaluations, regularization_terms))

    @pytest.fixture(params=[True, False])
    def progression(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def regularization(self, request):
        return request.param

    def test_if_gradient_ascent_is_correctly_called(self, config, ascending_func):
        methods.ascend("model", "initial_guess", config, ascending_func=ascending_func)
        ascending_func.assert_called_once_with("model", "initial_guess", key1="value1", key2="value2")

    def test_if_result_is_correctly_returned(self, config, ascending_func, progression, regularization):
        expected_output = dict(
            function_evaluations=["initial_evaluation", "intermittent_evaluation", "final_evaluation"]
        )
        if progression:
            expected_output["progression"] = ["initial_mei", "intermittent_mei", "final_mei"]
        if regularization:
            expected_output["regularization_terms"] = ["initial_term", "intermittent_term", "final_term"]
        mei, score, output = methods.ascend("model", "initial_guess", config, ascending_func=ascending_func)
        assert mei == "final_mei"
        assert score == "final_evaluation"
        assert output == expected_output


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
