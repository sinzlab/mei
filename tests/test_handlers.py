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
