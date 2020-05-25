from unittest.mock import MagicMock
from functools import partial

import pytest

from mei import mixins


@pytest.fixture
def mei_method(insert1):
    mei_method = mixins.MEIMethodMixin
    mei_method.insert1 = insert1
    return mei_method


@pytest.fixture
def method_fn():
    return MagicMock(name="method_fn", return_value=("mei", "score", "output"))


def test_that_method_is_correctly_inserted(mei_method, insert1):
    method_config = MagicMock(name="method_config")
    mei_method().add_method("method_fn", method_config)
    insert1.assert_called_once_with(
        dict(
            method_fn="method_fn",
            method_hash="d41d8cd98f00b204e9800998ecf8427e",
            method_config=method_config,
            method_comment="",
        )
    )


class TestGenerateMEI:
    @pytest.fixture
    def generate_mei(self, mei_method, dataloaders, model, seed):
        return partial(mei_method().generate_mei, dataloaders, model, dict(key="key"), seed)

    @pytest.fixture
    def mei_method(self, mei_method, magic_and, import_func):
        mei_method.__and__ = magic_and
        mei_method.import_func = import_func
        return mei_method

    @pytest.fixture
    def dataloaders(self):
        return MagicMock(name="dataloaders")

    @pytest.fixture
    def model(self):
        return MagicMock(name="model")

    @pytest.fixture
    def seed(self):
        return 42

    @pytest.fixture
    def magic_and(self):
        magic_and = MagicMock()
        magic_and.return_value.fetch1.return_value = "method_fn", "method_config"
        return magic_and

    @pytest.fixture
    def import_func(self, method_fn):
        return MagicMock(name="import_func", return_value=method_fn)

    def test_that_method_is_correctly_fetched(self, generate_mei, magic_and):
        generate_mei()
        magic_and.assert_called_once_with(dict(key="key"))
        magic_and.return_value.fetch1.assert_called_once_with("method_fn", "method_config")

    def test_if_method_function_is_correctly_imported(self, generate_mei, import_func):
        generate_mei()
        import_func.assert_called_once_with("method_fn")

    def test_if_method_function_is_correctly_called(self, generate_mei, model, dataloaders, seed, method_fn):
        generate_mei()
        method_fn.assert_called_once_with(dataloaders, model, "method_config", seed)

    def test_if_returned_mei_entity_is_correct(self, generate_mei):
        mei_entity = generate_mei()
        assert mei_entity == dict(key="key", mei="mei", score="score", output="output")
