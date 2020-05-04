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
        config,
        get_dims,
        create_initial_guess,
        input_cls,
        mei_class,
        import_func,
        optimize_func,
        tracker_cls,
    ):
        def _gradient_ascent(
            use_transform=False, use_regularization=False, use_precondition=False, use_postprocessing=False
        ):
            return partial(
                methods.gradient_ascent,
                dataloaders,
                model,
                config(
                    use_transform=use_transform,
                    use_regularization=use_regularization,
                    use_precondition=use_precondition,
                    use_postprocessing=use_postprocessing,
                ),
                42,
                get_dims=get_dims,
                create_initial_guess=create_initial_guess,
                input_cls=input_cls,
                mei_class=mei_class,
                import_func=import_func,
                optimize_func=optimize_func,
                tracker_cls=tracker_cls,
            )

        return _gradient_ascent

    @pytest.fixture
    def dataloaders(self):
        return dict(train="train_dataloaders")

    @pytest.fixture
    def model(self):
        return MagicMock(name="model")

    @pytest.fixture
    def config(self):
        def _config(use_transform=False, use_regularization=False, use_precondition=False, use_postprocessing=False):
            config = dict(
                device="cpu",
                optimizer=dict(path="optimizer_path", kwargs=dict(optimizer_kwarg1=0, optimizer_kwarg2=1)),
                stopper=dict(path="stopper_path", kwargs=dict(stopper_kwarg1=0, stopper_kwarg2=1)),
                objectives=[
                    dict(path="obj1_path", kwargs=dict(obj1_kwarg1=0, obj1_kwarg2=1)),
                    dict(path="obj2_path", kwargs=dict(obj2_kwarg1=0, obj2_kwarg2=1)),
                ],
            )
            if use_transform:
                config = dict(
                    config, transform=dict(path="transform_path", kwargs=dict(transform_kwarg1=0, transform_kwarg2=1))
                )
            else:
                config = dict(config, transform=None)
            if use_regularization:
                config = dict(
                    config,
                    regularization=dict(
                        path="regularization_path", kwargs=dict(regularization_kwarg1=0, regularization_kwarg2=1)
                    ),
                )
            else:
                config = dict(config, regularization=None)
            if use_precondition:
                config = dict(
                    config,
                    precondition=dict(
                        path="precondition_path", kwargs=dict(precondition_kwarg1=0, precondition_kwarg2=1)
                    ),
                )
            else:
                config = dict(config, precondition=None)
            if use_postprocessing:
                config = dict(
                    config,
                    postprocessing=dict(
                        path="postprocessing_path", kwargs=dict(postprocessing_kwarg1=0, postprocessing_kwarg2=1)
                    ),
                )
            else:
                config = dict(config, postprocessing=None)
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
            use_transform=False, use_regularization=False, use_precondition=False, use_postprocessing=False
        ):
            import_func_calls = [
                call("optimizer_path", dict(params=["initial_guess"], optimizer_kwarg1=0, optimizer_kwarg2=1)),
                call("stopper_path", dict(stopper_kwarg1=0, stopper_kwarg2=1)),
                call("obj1_path", dict(obj1_kwarg1=0, obj1_kwarg2=1)),
                call("obj2_path", dict(obj2_kwarg1=0, obj2_kwarg2=1)),
            ]
            if use_transform:
                import_func_calls.append(call("transform_path", dict(transform_kwarg1=0, transform_kwarg2=1)))
            if use_regularization:
                import_func_calls.append(
                    call("regularization_path", dict(regularization_kwarg1=0, regularization_kwarg2=1))
                )
            if use_precondition:
                import_func_calls.append(call("precondition_path", dict(precondition_kwarg1=0, precondition_kwarg2=1)))
            if use_postprocessing:
                import_func_calls.append(
                    call("postprocessing_path", dict(postprocessing_kwarg1=0, postprocessing_kwarg2=1))
                )
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

    def test_if_seed_is_set(self, gradient_ascent):
        set_seed = MagicMock(name="set_seed")
        gradient_ascent(use_transform=True)(set_seed=set_seed)
        set_seed.assert_called_once_with(42)

    def test_model_is_switched_to_eval_mode(self, gradient_ascent, model):
        gradient_ascent(use_transform=True)()
        model.eval.assert_called_once_with()

    def test_if_model_is_switched_to_device(self, gradient_ascent, model):
        gradient_ascent(use_transform=True)()
        model.to.assert_called_once_with("cpu")

    def test_if_get_dims_is_correctly_called(self, gradient_ascent, get_dims):
        gradient_ascent(use_transform=True)()
        get_dims.assert_called_once_with("train_dataloaders")

    def test_if_create_initial_guess_is_correctly_called(self, gradient_ascent, create_initial_guess):
        gradient_ascent(use_transform=True)()
        create_initial_guess.assert_called_once_with(1, 5, 15, 15, device="cpu")

    def test_if_input_class_is_correctly_called(self, gradient_ascent, input_cls):
        gradient_ascent()()
        input_cls.assert_called_once_with("initial_guess")

    @pytest.mark.parametrize("use_transform", [True, False])
    @pytest.mark.parametrize("use_regularization", [True, False])
    @pytest.mark.parametrize("use_precondition", [True, False])
    @pytest.mark.parametrize("use_postprocessing", [True, False])
    def test_if_import_func_is_correctly_called(
        self,
        gradient_ascent,
        import_func,
        import_func_calls,
        use_transform,
        use_regularization,
        use_precondition,
        use_postprocessing,
    ):
        gradient_ascent(
            use_transform=use_transform,
            use_regularization=use_regularization,
            use_precondition=use_precondition,
            use_postprocessing=use_postprocessing,
        )()
        calls = import_func_calls(
            use_transform=use_transform,
            use_regularization=use_regularization,
            use_precondition=use_precondition,
            use_postprocessing=use_postprocessing,
        )
        assert import_func.mock_calls == calls

    def test_if_tracker_is_correctly_called(self, gradient_ascent, tracker_cls):
        gradient_ascent()()
        tracker_cls.assert_called_once_with(obj1_path="obj1", obj2_path="obj2")

    @pytest.mark.parametrize("use_transform", [True, False])
    @pytest.mark.parametrize("use_regularization", [True, False])
    @pytest.mark.parametrize("use_precondition", [True, False])
    @pytest.mark.parametrize("use_postprocessing", [True, False])
    def test_if_mei_is_correctly_initialized(
        self,
        gradient_ascent,
        model,
        mei_class,
        mei_class_call,
        use_transform,
        use_regularization,
        use_precondition,
        use_postprocessing,
    ):
        gradient_ascent(
            use_transform=use_transform,
            use_regularization=use_regularization,
            use_precondition=use_precondition,
            use_postprocessing=use_postprocessing,
        )()
        assert mei_class.mock_calls == [
            mei_class_call(
                use_transform=use_transform,
                use_regularization=use_regularization,
                use_precondition=use_precondition,
                use_postprocessing=use_postprocessing,
            )
        ]

    def test_if_optimize_func_is_correctly_called(self, gradient_ascent, optimize_func, tracker_instance):
        gradient_ascent(use_transform=True)()
        optimize_func.assert_called_once_with("mei", "stopper", tracker_instance)

    def test_if_result_is_returned(self, gradient_ascent):
        assert gradient_ascent(use_transform=True)() == ("final_evaluation", "mei", "tracker_log")
