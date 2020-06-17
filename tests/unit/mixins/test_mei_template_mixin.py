from unittest.mock import MagicMock, call

import pytest

from mei import mixins


@pytest.fixture
def mei_template(trained_model_table, model_loader_class):
    mei_template = mixins.MEITemplateMixin
    mei_template.trained_model_table = trained_model_table
    mei_template.model_loader_class = model_loader_class
    return mei_template


@pytest.fixture
def trained_model_table():
    return MagicMock(name="trained_model_table")


@pytest.fixture
def model_loader_class(model_loader):
    return MagicMock(name="model_loader_class", return_value=model_loader)


@pytest.fixture
def model_loader():
    model_loader = MagicMock(name="model_loader")
    model_loader.load.return_value = "dataloaders", "model"
    return model_loader


def test_if_model_loader_is_correctly_initialized(mei_template, trained_model_table, model_loader_class):
    mei_template(cache_size_limit=5)
    model_loader_class.assert_called_once_with(trained_model_table, cache_size_limit=5)


class TestMake:
    @pytest.fixture
    def mei_template(self, mei_template, selector_table, method_table, seed_table, insert1, save, model_loader_class):
        mei_template.selector_table = selector_table
        mei_template.method_table = method_table
        mei_template.seed_table = seed_table
        mei_template.insert1 = insert1
        mei_template.save = save
        mei_template.model_loader_class = model_loader_class
        get_temp_dir = MagicMock(name="get_temp_dir")
        get_temp_dir.return_value.__enter__.return_value = "/temp_dir"
        mei_template.get_temp_dir = get_temp_dir
        mei_template._create_random_filename = MagicMock(
            name="create_random_filename", side_effect=["filename1", "filename2"]
        )
        return mei_template

    @pytest.fixture
    def selector_table(self):
        selector_table = MagicMock(name="selector_table")
        selector_table.return_value.get_output_selected_model.return_value = "output_selected_model"
        return selector_table

    @pytest.fixture
    def method_table(self):
        method_table = MagicMock(name="method_table")
        method_table.return_value.generate_mei.return_value = dict(mei="mei", output="output")
        return method_table

    @pytest.fixture
    def seed_table(self):
        seed_table = MagicMock(name="seed_table")
        seed_table.return_value.__and__.return_value.fetch1.return_value = "seed"
        return seed_table

    @pytest.fixture
    def save(self):
        return MagicMock(name="save")

    def test_if_model_is_correctly_loaded(self, key, mei_template, model_loader):
        mei_template().make(key)
        model_loader.load.assert_called_once_with(key=key)

    def test_if_correct_model_output_is_selected(self, key, mei_template, selector_table):
        mei_template().make(key)
        selector_table.return_value.get_output_selected_model.assert_called_once_with("model", key)

    def test_if_seed_is_correctly_fetched(self, key, mei_template, seed_table):
        mei_template().make(key)
        seed_table.return_value.__and__.assert_called_once_with(key)
        seed_table.return_value.__and__.return_value.fetch1.assert_called_once_with("mei_seed")

    def test_if_mei_is_correctly_generated(self, key, mei_template, method_table):
        mei_template().make(key)
        method_table.return_value.generate_mei.assert_called_once_with(
            "dataloaders", "output_selected_model", key, "seed"
        )

    def test_if_mei_is_correctly_saved(self, key, mei_template, save):
        mei_template().make(key)
        assert save.call_count == 2
        save.has_calls(
            call("mei", "/temp_dir/mei_filename1.pth.tar"), call("output", "/temp_dir/output_filename2.pth.tar")
        )

    def test_if_mei_entity_is_correctly_saved(self, key, mei_template, insert1):
        mei_template().make(key)
        insert1.assert_called_once_with(
            dict(mei="/temp_dir/mei_filename1.pth.tar", output="/temp_dir/output_filename2.pth.tar")
        )
