from unittest.mock import MagicMock

import pytest

from mei import mixins


@pytest.fixture
def objective_template(dataset_table):
    objective_template = mixins.CSRFV1ObjectiveTemplateMixin
    objective_template.dataset_table = dataset_table
    objective_template.dataset_fn = "dataset_fn"
    return objective_template


@pytest.fixture
def dataset_table():
    return MagicMock(name="dataset_table")


def test_if_key_source_is_correct(objective_template, dataset_table):
    dataset_table.return_value.__and__.return_value = "key_source"
    assert objective_template()._key_source == "key_source"
    dataset_table.return_value.__and__.assert_called_once_with(dict(dataset_fn="dataset_fn"))


class TestMake:
    @pytest.fixture
    def objective_template(self, objective_template, dataset_table, insert):
        objective_template.insert = insert
        return objective_template

    @pytest.fixture
    def dataset_table(self, dataset_table):
        dataset_table.return_value.__and__.return_value.fetch1.return_value = "dataset_config"
        return dataset_table

    @pytest.fixture
    def insert(self):
        return MagicMock(name="insert")

    @pytest.fixture
    def get_mappings(self):
        return MagicMock(return_value="mappings")

    def test_if_dataset_config_is_correctly_fetched(self, key, objective_template, dataset_table, get_mappings):
        objective_template().make(key, get_mappings=get_mappings)
        dataset_table.return_value.__and__.assert_called_once_with(key)
        dataset_table.return_value.__and__.return_value.fetch1.assert_called_once_with("dataset_config")

    def test_if_get_mappings_is_correctly_called(self, key, objective_template, get_mappings):
        objective_template().make(key, get_mappings=get_mappings)
        get_mappings.assert_called_once_with("dataset_config", key)

    def test_if_mappings_are_correctly_inserted(self, key, objective_template, insert, get_mappings):
        objective_template().make(key, get_mappings=get_mappings)
        insert.assert_called_once_with("mappings")


class TestGetOutputSelectedModel:
    @pytest.fixture
    def objective_template(self, objective_template, constrained_output_model, magic_and):
        objective_template.constrained_output_model = constrained_output_model
        objective_template.__and__ = magic_and
        return objective_template

    @pytest.fixture
    def constrained_output_model(self):
        return MagicMock(name="constrained_output_model", return_value="constrained_output_model")

    @pytest.fixture
    def magic_and(self):
        magic_and = MagicMock()
        magic_and.return_value.fetch1.return_value = "neuron_pos", "session_id"
        return magic_and

    @pytest.fixture
    def model(self):
        return MagicMock(name="model")

    def test_if_neuron_position_and_session_id_are_correctly_fetched(self, key, model, objective_template, magic_and):
        objective_template().get_output_selected_model(model, key)
        magic_and.assert_called_once_with(key)
        magic_and.return_value.fetch1.assert_called_once_with("neuron_position", "session_id")

    def test_if_constrained_output_model_is_correctly_initialized(
        self, key, model, objective_template, constrained_output_model
    ):
        objective_template().get_output_selected_model(model, key)
        constrained_output_model.assert_called_once_with(
            model, "neuron_pos", forward_kwargs=dict(data_key="session_id")
        )

    def test_if_output_selected_model_is_correctly_returned(self, key, model, objective_template):
        output_selected_model = objective_template().get_output_selected_model(model, key)
        assert output_selected_model == "constrained_output_model"
