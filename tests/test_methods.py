from unittest.mock import MagicMock, call
from functools import partial

import pytest

from featurevis import methods


class TestAscend:
    @pytest.fixture
    def config(self):
        return dict(key1="value1", key2="value2")

    @pytest.fixture
    def ascending_func(self, mei, regularization):
        function_evaluations = ["initial_evaluation", "intermittent_evaluation", "final_evaluation"]
        if regularization:
            regularization_terms = ["initial_term", "intermittent_term", "final_term"]
        else:
            regularization_terms = []
        return MagicMock(return_value=(mei, function_evaluations, regularization_terms))

    @pytest.fixture
    def mei(self, final_mei, progression):
        if progression:
            initial_mei = MagicMock()
            initial_mei.cpu.return_value.squeeze.return_value = "initial_mei"
            intermittent_mei = MagicMock()
            intermittent_mei.cpu.return_value.squeeze.return_value = "intermittent_mei"
            return [initial_mei, intermittent_mei, final_mei]
        else:
            return final_mei

    @pytest.fixture(params=[True, False])
    def regularization(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def progression(self, request):
        return request.param

    @pytest.fixture
    def final_mei(self):
        final_mei = MagicMock(name="final_mei")
        final_mei.cpu.return_value.squeeze.return_value = "final_mei"
        return final_mei

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

    def test_if_mei_is_transferred_to_cpu(self, config, ascending_func, progression, mei, final_mei):
        methods.ascend("model", "initial_guess", config, ascending_func=ascending_func)
        final_mei.cpu.assert_called_once_with()
        if progression:
            meis = mei
            for mei in meis:
                mei.cpu.assert_called_once_with()


class TestGradientAscent:
    @pytest.fixture
    def gradient_ascent(
        self, dataloaders, model, config, set_seed, import_object, get_dims, get_initial_guess, ascend_func
    ):
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
            ascend_func=ascend_func,
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
    def ascend_func(self):
        return MagicMock(return_value=("mei", "score", "output"))

    def test_if_import_object_is_correctly_called(self, gradient_ascent, import_object):
        gradient_ascent()
        import_object.assert_called_once_with("module.function")

    def test_if_get_dims_is_correctly_called(self, gradient_ascent, get_dims):
        gradient_ascent()
        get_dims.assert_called_once_with(dict(session1=None))

    def test_if_get_initial_guess_is_correctly_called(self, gradient_ascent, get_initial_guess):
        gradient_ascent()
        get_initial_guess.assert_called_once_with(1, 10, 24, 24, device="device")

    def test_if_ascend_is_correctly_called(self, gradient_ascent, ascend_func, model, initial_guess):
        gradient_ascent()
        ascend_func.assert_called_once_with(
            model,
            initial_guess,
            dict(
                optim_kwargs=dict(),
                transform=None,
                regularization="imported_function",
                gradient_f=None,
                post_update=None,
            ),
        )

    def test_if_mei_and_evaluations_are_returned(self, gradient_ascent):
        assert gradient_ascent() == ("mei", "score", "output")

    def test_if_seed_is_correctly_set(self, gradient_ascent, set_seed):
        gradient_ascent()
        set_seed.assert_called_once_with("seed")

    def test_if_eval_method_of_model_is_called(self, gradient_ascent, model):
        gradient_ascent()
        model.eval.assert_called_once_with()

    def test_if_model_is_transferred_to_device(self, gradient_ascent, model):
        gradient_ascent()
        model.to.assert_called_once_with("device")


class TestAscendGradient:
    @pytest.fixture
    def ascend_gradient(
        self, dataloaders, model, config, get_dims, create_initial_guess, mei_class, resolve_func, optimize_func
    ):
        return partial(
            methods.ascend_gradient,
            dataloaders,
            model,
            config,
            42,
            get_dims=get_dims,
            create_initial_guess=create_initial_guess,
            mei_class=mei_class,
            resolve_func=resolve_func,
            optimize_func=optimize_func,
        )

    @pytest.fixture
    def dataloaders(self):
        return dict(train="train_dataloaders")

    @pytest.fixture
    def model(self):
        return MagicMock(name="model")

    @pytest.fixture
    def config(self):
        return dict(
            device="cpu",
            optimizer="optimizer",
            optimizer_kwargs=dict(optimizer_kwarg1=0, optimizer_kwarg2=1),
            checker="checker",
            checker_kwargs=dict(checker_kwarg1=0, checker_kwarg2=1),
        )

    @pytest.fixture
    def get_dims(self):
        return MagicMock(name="get_dims", return_value=dict(dl1=dict(inputs=(10, 5, 15, 15))))

    @pytest.fixture
    def create_initial_guess(self):
        return MagicMock(name="create_initial_guess", return_value="initial_guess")

    @pytest.fixture
    def mei_class(self):
        return MagicMock(name="mei_class", return_value="mei")

    @pytest.fixture
    def resolve_func(self, optimizer_class, checker_class):
        return MagicMock(name="resolve_func", side_effect=[optimizer_class, checker_class])

    @pytest.fixture
    def optimizer_class(self):
        return MagicMock(name="optimizer_class", return_value="optimizer")

    @pytest.fixture
    def checker_class(self):
        return MagicMock(name="checker_class", return_value="checker")

    @pytest.fixture
    def optimize_func(self):
        return MagicMock(name="optimize_func", return_value=("mei", "final_evaluation"))

    def test_if_seed_is_set(self, ascend_gradient):
        set_seed = MagicMock(name="set_seed")
        ascend_gradient(set_seed=set_seed)
        set_seed.assert_called_once_with(42)

    def test_model_is_switched_to_eval_mode(self, ascend_gradient, model):
        ascend_gradient()
        model.eval.assert_called_once_with()

    def test_if_model_is_switched_to_device(self, ascend_gradient, model):
        ascend_gradient()
        model.to.assert_called_once_with("cpu")

    def test_if_get_dims_is_correctly_called(self, ascend_gradient, get_dims):
        ascend_gradient()
        get_dims.assert_called_once_with("train_dataloaders")

    def test_if_create_initial_guess_is_correctly_called(self, ascend_gradient, create_initial_guess):
        ascend_gradient()
        create_initial_guess.assert_called_once_with(1, 5, 15, 15, device="cpu")

    def test_if_mei_is_correctly_initialized(self, ascend_gradient, model, mei_class):
        ascend_gradient()
        mei_class.assert_called_once_with(model, "initial_guess")

    def test_if_resolve_func_is_correctly_called(self, ascend_gradient, resolve_func):
        ascend_gradient()
        resolve_func.assert_has_calls([call("optimizer", "torch.optim"), call("checker", "featurevis.checkers")])

    def test_if_optimizer_is_correctly_initialized(self, ascend_gradient, optimizer_class):
        ascend_gradient()
        optimizer_class.assert_called_once_with(["initial_guess"], optimizer_kwarg1=0, optimizer_kwarg2=1)

    def test_if_checker_is_correctly_initialized(self, ascend_gradient, checker_class):
        ascend_gradient()
        checker_class.assert_called_once_with(checker_kwarg1=0, checker_kwarg2=1)

    def test_if_optimize_func_is_correctly_called(self, ascend_gradient, optimize_func):
        ascend_gradient()
        optimize_func.assert_called_once_with("mei", "optimizer", "checker")

    def test_if_result_is_returned(self, ascend_gradient):
        assert ascend_gradient() == ("mei", "final_evaluation", dict())
