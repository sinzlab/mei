from unittest.mock import MagicMock
from contextlib import contextmanager

import pytest

from featurevis import handlers


@contextmanager
def does_not_raise():
    yield


class TestMEIMethodHandler:
    def test_that_call_to_insert_method_is_correct(self):
        facade = MagicMock()

        handler = handlers.MEIMethodHandler(facade)
        handler.add_method("method_fn", "method_config")

        facade.insert_method.assert_called_once_with(
            dict(method_fn="method_fn", method_hash="57f270bf813f42465bd9c21a364bdb2b", method_config="method_config")
        )

    @pytest.fixture
    def facade(self):
        facade = MagicMock()
        facade.fetch_method.return_value = "method_fn", "method_config"
        return facade

    @pytest.fixture
    def import_func(self):
        import_func = MagicMock()
        import_func.return_value.return_value = "mei", "evaluations"
        return import_func

    @pytest.fixture
    def handler(self, facade):
        return handlers.MEIMethodHandler(facade)

    @pytest.fixture
    def mei(self, handler, import_func):
        return handler.generate_mei("dataloader", "model", dict(key="key"), import_func=import_func)

    def test_that_call_to_fetch_method_is_correct(self, mei, facade):
        facade.fetch_method.assert_called_once_with(dict(key="key"))

    def test_that_import_func_is_called_correctly(self, mei, import_func):
        import_func.assert_called_once_with("method_fn")

    def test_that_method_func_is_called_correctly(self, mei, import_func):
        import_func.return_value.assert_called_once_with("dataloader", "model", "method_config")

    def test_that_generated_mei_is_returned(self, mei, import_func):
        assert mei == dict(key="key", evaluations="evaluations", mei="mei")


class TestMEIHandler:
    @pytest.fixture
    def facade(self, mei_entity):
        facade = MagicMock()
        facade.trained_model_table = "trained_model_table_instance"
        facade.get_output_selected_model.return_value = "output_selected_model"
        facade.generate_mei.return_value = mei_entity
        return facade

    @pytest.fixture
    def model_loader(self):
        model_loader = MagicMock()
        model_loader.return_value.load.return_value = "dataloaders", "model"
        return model_loader

    @pytest.fixture
    def temp_dir_func(self):
        temp_dir_func = MagicMock()
        temp_dir_func.return_value.__enter__.return_value = "/temp_dir"
        return temp_dir_func

    @pytest.fixture
    def mei_entity(self):
        mei = MagicMock()
        mei.squeeze.return_value = "mei"
        return dict(mei=mei)

    @pytest.fixture
    def save_func(self):
        return MagicMock()

    @pytest.fixture
    def handler(self, facade, model_loader, temp_dir_func, save_func):
        handler = handlers.MEIHandler(facade, model_loader=model_loader, cache_size_limit=5)
        handler.make("key", save_func=save_func, temp_dir_func=temp_dir_func)
        return handler

    def test_that_model_loader_is_correctly_initialized(self, model_loader, handler):
        model_loader.assert_called_once_with("trained_model_table_instance", cache_size_limit=5)

    def test_that_call_to_load_is_correct(self, handler, model_loader):
        model_loader.return_value.load.assert_called_once_with(key="key")

    def test_that_call_to_get_output_selected_model_is_correct(self, facade, handler):
        facade.get_output_selected_model.assert_called_once_with("model", "key")

    def test_that_call_to_generate_mei_is_correct(self, facade, handler):
        facade.generate_mei.assert_called_once_with("dataloaders", "output_selected_model", "key")

    def test_that_call_to_save_func_is_correct(self, handler, save_func):
        save_func.assert_called_once_with("mei", "/temp_dir/d41d8cd98f00b204e9800998ecf8427e.pth.tar")

    def test_that_call_to_insert_mei_is_correct(self, facade, handler):
        facade.insert_mei.assert_called_once_with(dict(mei="/temp_dir/d41d8cd98f00b204e9800998ecf8427e.pth.tar"))
