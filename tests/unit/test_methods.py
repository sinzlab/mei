from unittest.mock import MagicMock, call
from functools import partial
from typing import Type

import pytest

from featurevis import methods
from featurevis.domain import Input
from featurevis.tracking import Tracker


class TestGradientAscent:
    @pytest.fixture
    def gradient_ascent(
        self,
        dataloaders,
        model,
        get_dims,
        create_initial_guess,
        input_cls,
        mei_class,
        import_func,
        optimize_func,
        tracker_cls,
    ):
        return partial(
            methods.gradient_ascent,
            dataloaders=dataloaders,
            model=model,
            seed=42,
            get_dims=get_dims,
            create_initial_guess=create_initial_guess,
            input_cls=input_cls,
            mei_class=mei_class,
            import_func=import_func,
            optimize_func=optimize_func,
            tracker_cls=tracker_cls,
        )

    @pytest.fixture
    def dataloaders(self):
        return dict(train="train_dataloaders")

    @pytest.fixture
    def model(self):
        return MagicMock(name="model")

    @pytest.fixture
    def config(self):
        def _config(
            n_kwargs=None,
            n_objectives=None,
            use_transform=False,
            use_regularization=False,
            use_precondition=False,
            use_postprocessing=False,
        ):
            def get_component_config(name):
                component_config = dict(path=name + "_path")
                if n_kwargs is not None:
                    component_config["kwargs"] = {name + "_kwarg" + str(i): i - 1 for i in range(1, n_kwargs + 1)}
                return component_config

            config = dict(
                device="cpu", optimizer=get_component_config("optimizer"), stopper=get_component_config("stopper")
            )
            if n_objectives is not None:
                objectives = [get_component_config("obj" + str(i)) for i in range(1, n_objectives + 1)]
                config = dict(config, objectives=objectives)
            if use_transform:
                config = dict(config, transform=get_component_config("transform"))
            if use_regularization:
                config = dict(config, regularization=get_component_config("regularization"))
            if use_precondition:
                config = dict(config, precondition=get_component_config("precondition"))
            if use_postprocessing:
                config = dict(config, postprocessing=get_component_config("postprocessing"))
            return config

        return _config

    @pytest.fixture
    def get_dims(self):
        return MagicMock(name="get_dims", return_value=dict(dl1=dict(inputs=(10, 5, 15, 15))))

    @pytest.fixture
    def create_initial_guess(self):
        return MagicMock(name="create_initial_guess", return_value="initial_guess")

    @pytest.fixture
    def input_cls(self):
        return MagicMock(name="input_cls", return_value="input_instance", spec=Input)

    @pytest.fixture
    def mei_class(self):
        return MagicMock(name="mei_class", return_value="mei")

    @pytest.fixture
    def import_func(self):
        def _import_func(name, _kwargs):
            return name.split("_")[0]

        return MagicMock(name="import_func", side_effect=_import_func)

    @pytest.fixture
    def optimize_func(self):
        return MagicMock(name="optimize_func", return_value=("mei", "final_evaluation"))

    @pytest.fixture
    def tracker_cls(self, tracker_instance):
        return MagicMock(name="tracker_cls", spec=Type[Tracker], return_value=tracker_instance)

    @pytest.fixture
    def tracker_instance(self):
        tracker = MagicMock(name="tracker_instance", spec=Tracker)
        tracker.log = "tracker_log"
        return tracker

    @pytest.fixture
    def import_func_calls(self):
        def _import_func_calls(
            n_kwargs=None,
            n_objectives=None,
            use_transform=False,
            use_regularization=False,
            use_precondition=False,
            use_postprocessing=False,
        ):
            def get_kwargs(name):
                if n_kwargs is None:
                    return dict()
                else:
                    return {name + "_kwarg" + str(i): i - 1 for i in range(1, n_kwargs + 1)}

            import_func_calls = [
                call("optimizer_path", dict(params=["initial_guess"], **get_kwargs("optimizer"))),
                call("stopper_path", get_kwargs("stopper")),
            ]
            if n_objectives is not None:
                import_func_calls.extend(
                    [call(f"obj{i}_path", get_kwargs(f"obj{i}")) for i in range(1, n_objectives + 1)]
                )
            if use_transform:
                import_func_calls.append(call("transform_path", get_kwargs("transform")))
            if use_regularization:
                import_func_calls.append(call("regularization_path", get_kwargs("regularization")))
            if use_precondition:
                import_func_calls.append(call("precondition_path", get_kwargs("precondition")))
            if use_postprocessing:
                import_func_calls.append(call("postprocessing_path", get_kwargs("postprocessing")))
            return import_func_calls

        return _import_func_calls

    @pytest.fixture
    def mei_class_call(self, model):
        def _mei_class_call(
            use_transform=False, use_regularization=False, use_precondition=False, use_postprocessing=False
        ):
            args = (model, "input_instance", "optimizer")
            kwargs = {}
            if use_transform:
                kwargs["transform"] = "transform"
            if use_regularization:
                kwargs["regularization"] = "regularization"
            if use_precondition:
                kwargs["precondition"] = "precondition"
            if use_postprocessing:
                kwargs["postprocessing"] = "postprocessing"
            return call(*args, **kwargs)

        return _mei_class_call

    def test_if_seed_is_set(self, gradient_ascent, config):
        set_seed = MagicMock(name="set_seed")
        gradient_ascent(config=config(), set_seed=set_seed)
        set_seed.assert_called_once_with(42)

    def test_model_is_switched_to_eval_mode(self, gradient_ascent, model, config):
        gradient_ascent(config=config())
        model.eval.assert_called_once_with()

    def test_if_model_is_switched_to_device(self, gradient_ascent, model, config):
        gradient_ascent(config=config())
        model.to.assert_called_once_with("cpu")

    def test_if_get_dims_is_correctly_called(self, gradient_ascent, config, get_dims):
        gradient_ascent(config=config())
        get_dims.assert_called_once_with("train_dataloaders")

    def test_if_create_initial_guess_is_correctly_called(self, gradient_ascent, config, create_initial_guess):
        gradient_ascent(config=config())
        create_initial_guess.assert_called_once_with(1, 5, 15, 15, device="cpu")

    def test_if_input_class_is_correctly_called(self, gradient_ascent, config, input_cls):
        gradient_ascent(config=config())
        input_cls.assert_called_once_with("initial_guess")

    @pytest.mark.parametrize("n_kwargs", [0, 1, 10])
    @pytest.mark.parametrize("n_objectives", [0, 1, 10])
    @pytest.mark.parametrize("use_transform", [True, False])
    @pytest.mark.parametrize("use_regularization", [True, False])
    @pytest.mark.parametrize("use_precondition", [True, False])
    @pytest.mark.parametrize("use_postprocessing", [True, False])
    def test_if_import_func_is_correctly_called(
        self,
        gradient_ascent,
        config,
        import_func,
        import_func_calls,
        n_kwargs,
        n_objectives,
        use_transform,
        use_regularization,
        use_precondition,
        use_postprocessing,
    ):
        gradient_ascent(
            config=config(
                n_kwargs=n_kwargs,
                n_objectives=n_objectives,
                use_transform=use_transform,
                use_regularization=use_regularization,
                use_precondition=use_precondition,
                use_postprocessing=use_postprocessing,
            )
        )
        calls = import_func_calls(
            n_kwargs=n_kwargs,
            n_objectives=n_objectives,
            use_transform=use_transform,
            use_regularization=use_regularization,
            use_precondition=use_precondition,
            use_postprocessing=use_postprocessing,
        )
        assert import_func.mock_calls == calls

    def test_if_import_func_is_correctly_called_if_objectives_is_none(
        self, gradient_ascent, config, import_func, import_func_calls
    ):
        config = config(n_objectives=None)
        gradient_ascent(config=config)
        assert import_func.mock_calls == import_func_calls()

    def test_if_import_func_is_correctly_called_if_kwargs_are_none(
        self, gradient_ascent, config, import_func, import_func_calls
    ):
        gradient_ascent(config=config(n_objectives=None))
        assert import_func.mock_calls == import_func_calls()

    @pytest.mark.parametrize("n_objectives", [0, 1, 10])
    def test_if_tracker_is_correctly_called(self, gradient_ascent, config, tracker_cls, n_objectives):
        gradient_ascent(config=config(n_objectives=n_objectives))
        tracker_cls.assert_called_once_with(**{f"obj{i}_path": f"obj{i}" for i in range(1, n_objectives + 1)})

    @pytest.mark.parametrize("use_transform", [True, False])
    @pytest.mark.parametrize("use_regularization", [True, False])
    @pytest.mark.parametrize("use_precondition", [True, False])
    @pytest.mark.parametrize("use_postprocessing", [True, False])
    def test_if_mei_is_correctly_initialized(
        self,
        gradient_ascent,
        model,
        config,
        mei_class,
        mei_class_call,
        use_transform,
        use_regularization,
        use_precondition,
        use_postprocessing,
    ):
        gradient_ascent(
            config=config(
                use_transform=use_transform,
                use_regularization=use_regularization,
                use_precondition=use_precondition,
                use_postprocessing=use_postprocessing,
            )
        )
        assert mei_class.mock_calls == [
            mei_class_call(
                use_transform=use_transform,
                use_regularization=use_regularization,
                use_precondition=use_precondition,
                use_postprocessing=use_postprocessing,
            )
        ]

    def test_if_optimize_func_is_correctly_called(self, gradient_ascent, config, optimize_func, tracker_instance):
        gradient_ascent(config=config())
        optimize_func.assert_called_once_with("mei", "stopper", tracker_instance)

    def test_if_result_is_returned(self, gradient_ascent, config):
        assert gradient_ascent(config=config()) == ("final_evaluation", "mei", "tracker_log")
