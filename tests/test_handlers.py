from unittest.mock import MagicMock, call
from contextlib import contextmanager

import pytest
import torch

from featurevis import handlers


@contextmanager
def does_not_raise():
    yield


class TestTrainedEnsembleModelHandler:
    @pytest.fixture
    def facade(self):
        return MagicMock()

    @pytest.fixture
    def handler(self, facade):
        return handlers.TrainedEnsembleModelHandler(facade)

    @pytest.mark.parametrize(
        "is_sufficient,expectation", [(True, does_not_raise()), (False, pytest.raises(ValueError))]
    )
    def test_if_key_restrictiveness_is_checked(self, facade, handler, is_sufficient, expectation):
        facade.properly_restricts = MagicMock(return_value=is_sufficient)

        with expectation:
            handler.create_ensemble("key")

    def test_if_call_to_properly_restricts_is_correct(self, facade, handler):
        handler.create_ensemble("key")

        facade.properly_restricts.assert_called_once_with("key")

    def test_if_call_to_fetch_trained_models_primary_keys_is_correct(self, facade, handler):
        handler.create_ensemble("key")

        facade.fetch_trained_models_primary_keys.assert_called_once_with("key")

    def test_if_call_to_insert_ensemble_is_correct(self, facade, handler):
        facade.fetch_primary_dataset_key = MagicMock(return_value=dict(d1=1, d2=2))

        handler.create_ensemble("key")

        facade.insert_ensemble.assert_called_once_with(
            dict(d1=1, d2=2, ensemble_hash="d41d8cd98f00b204e9800998ecf8427e")
        )

    def test_if_call_to_insert_members_is_correct(self, facade, handler):
        facade.fetch_primary_dataset_key = MagicMock(return_value=dict(d=1))
        facade.fetch_trained_models_primary_keys = MagicMock(return_value=[dict(m=0), dict(m=1)])

        handler.create_ensemble("key")

        facade.insert_members.assert_called_once_with(
            [
                dict(d=1, ensemble_hash="536072017a2a3501ea8f09fffa51ee61", m=0),
                dict(d=1, ensemble_hash="536072017a2a3501ea8f09fffa51ee61", m=1),
            ]
        )

    def test_if_call_to_fetch_trained_models_is_correct(self, facade, handler):
        facade.fetch_trained_models = MagicMock(return_value=["m"])
        facade.load_model = MagicMock(return_value=("dataloader", MagicMock()))

        handler.load_model("key")

        facade.fetch_trained_models.assert_called_once_with("key")

    def test_if_call_to_load_model_is_correct(self, facade, handler):
        facade.fetch_trained_models = MagicMock(return_value=["m1", "m2"])
        facade.load_model = MagicMock(return_value=("dataloader", MagicMock()))

        handler.load_model("key")

        assert facade.load_model.call_count == 2
        facade.load_model.assert_has_calls([call(key="m1"), call(key="m2")])

    def test_that_models_are_switched_to_evaluation_mode(self, facade, handler):
        facade.fetch_trained_models = MagicMock(return_value=["m1", "m2"])
        model = MagicMock()
        facade.load_model = MagicMock(return_value=("dataloader", model))

        handler.load_model("key")

        assert model.eval.call_count == 2

    def test_that_ensemble_model_is_correctly_constructed(self, facade, handler):
        facade.fetch_trained_models = MagicMock(return_value=["m1", "m2"])
        outputs = (torch.tensor([4.0, 7.0]), torch.tensor([6.0, 8.0]))
        model = MagicMock(side_effect=outputs)
        facade.load_model = MagicMock(return_value=("dataloader", model))

        _, ensemble = handler.load_model("key")

        assert torch.allclose(ensemble("key"), torch.tensor([5, 7.5]))

    def test_that_only_first_dataloader_is_returned(self, facade, handler):
        facade.fetch_trained_models = MagicMock(return_value=["m1", "m2"])
        dataloaders = [("dataloader1", MagicMock()), ("dataloader2", MagicMock())]
        facade.load_model = MagicMock(side_effect=dataloaders)

        dataloader, _ = handler.load_model("key")

        assert dataloader == "dataloader1"


class TestMEIMethodHandler:
    def test_that_call_to_insert_method_is_correct(self):
        facade = MagicMock()

        handler = handlers.MEIMethodHandler(facade)
        handler.add_method("method_fn", "method_config")

        facade.insert_method.assert_called_once_with(
            dict(method_fn="method_fn", method_hash="57f270bf813f42465bd9c21a364bdb2b", method_config="method_config")
        )

    def test_that_call_to_fetch_method_is_correct(self):
        facade = MagicMock()
        facade.fetch_method.return_value = "method_fn", "method_config"

        import_func = MagicMock()
        import_func.return_value.return_value = "mei", "evaluations"

        handler = handlers.MEIMethodHandler(facade)
        handler.generate_mei("dataloader", "model", dict(key="key"), import_func=import_func)

        facade.fetch_method.assert_called_once_with(dict(key="key"))

    def test_that_import_func_is_called_correctly(self):
        facade = MagicMock()
        facade.fetch_method.return_value = "method_fn", "method_config"

        import_func = MagicMock()
        import_func.return_value.return_value = "mei", "evaluations"

        handler = handlers.MEIMethodHandler(facade)
        handler.generate_mei("dataloader", "model", dict(key="key"), import_func=import_func)

        import_func.assert_called_once_with("method_fn")

    def test_that_method_func_is_called_correctly(self):
        facade = MagicMock()
        facade.fetch_method.return_value = "method_fn", "method_config"

        import_func = MagicMock()
        import_func.return_value.return_value = "mei", "evaluations"

        handler = handlers.MEIMethodHandler(facade)
        handler.generate_mei("dataloader", "model", dict(key="key"), import_func=import_func)

        import_func.return_value.assert_called_once_with("dataloader", "model", "method_config")

    def test_that_generated_mei_is_returned(self):
        facade = MagicMock()
        facade.fetch_method.return_value = "method_fn", "method_config"

        import_func = MagicMock()
        import_func.return_value.return_value = "mei", "evaluations"

        handler = handlers.MEIMethodHandler(facade)
        mei = handler.generate_mei("dataloader", "model", dict(key="key"), import_func=import_func)

        assert mei == dict(key="key", evaluations="evaluations", mei="mei")


class TestMEIHandler:
    def test_that_model_loader_is_correctly_initialized(self):
        model_loader = MagicMock()
        facade = MagicMock()
        facade.trained_model_table = "trained_model_table_instance"

        _ = handlers.MEIHandler(facade, model_loader=model_loader, cache_size_limit=5)

        model_loader.assert_called_once_with("trained_model_table_instance", cache_size_limit=5)

    def test_that_call_to_load_is_correct(self):
        model_loader_instance = MagicMock()
        model_loader_instance.load.return_value = "dataloaders", "model"
        model_loader = MagicMock(return_value=model_loader_instance)

        handler = handlers.MEIHandler(MagicMock(), model_loader=model_loader)
        handler.make("key", save_func=MagicMock())

        model_loader_instance.load.assert_called_once_with(key="key")

    def test_that_call_to_get_output_selected_model_is_correct(self):
        model_loader_instance = MagicMock()
        model_loader_instance.load.return_value = ("dataloaders", "model")
        model_loader = MagicMock(return_value=model_loader_instance)
        facade = MagicMock()

        handler = handlers.MEIHandler(facade, model_loader=model_loader)
        handler.make("key", save_func=MagicMock())

        facade.get_output_selected_model.assert_called_once_with("model", "key")

    def test_that_call_to_generate_mei_is_correct(self):
        model_loader_instance = MagicMock()
        model_loader_instance.load.return_value = ("dataloaders", "model")
        model_loader = MagicMock(return_value=model_loader_instance)
        facade = MagicMock()
        facade.get_output_selected_model.return_value = "output_selected_model"

        handler = handlers.MEIHandler(facade, model_loader=model_loader)
        handler.make("key", save_func=MagicMock())

        facade.generate_mei.assert_called_once_with("dataloaders", "output_selected_model", "key")

    def test_that_call_to_save_func_is_correct(self):
        model_loader_instance = MagicMock()
        model_loader_instance.load.return_value = ("dataloaders", "model")
        model_loader = MagicMock(return_value=model_loader_instance)

        mei = MagicMock()
        mei.squeeze.return_value = "mei"
        mei_entity = MagicMock()
        mei_entity.pop.return_value = mei

        facade = MagicMock()
        facade.generate_mei.return_value = mei_entity

        temp_dir_func = MagicMock()
        temp_dir_func.return_value.__enter__.return_value = "/temp_dir"

        save_func = MagicMock()

        handler = handlers.MEIHandler(facade, model_loader=model_loader)
        handler.make("key", save_func=save_func, temp_dir_func=temp_dir_func)

        save_func.assert_called_once_with("mei", "/temp_dir/d41d8cd98f00b204e9800998ecf8427e.pth.tar")

    def test_that_call_to_insert_mei_is_correct(self):
        model_loader_instance = MagicMock()
        model_loader_instance.load.return_value = ("dataloaders", "model")
        model_loader = MagicMock(return_value=model_loader_instance)

        facade = MagicMock()
        facade.generate_mei.return_value = dict(mei=torch.tensor([1, 2, 3]))

        temp_dir_func = MagicMock()
        temp_dir_func.return_value.__enter__.return_value = "/temp_dir"

        handler = handlers.MEIHandler(facade, model_loader=model_loader)
        handler.make("key", save_func=MagicMock(), temp_dir_func=temp_dir_func)

        facade.insert_mei.assert_called_once_with(dict(mei="/temp_dir/d41d8cd98f00b204e9800998ecf8427e.pth.tar"))
