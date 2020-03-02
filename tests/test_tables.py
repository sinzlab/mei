from unittest.mock import MagicMock, call
from contextlib import contextmanager

import pytest
import torch

from featurevis import tables


@contextmanager
def does_not_raise():
    yield


class TestTrainedEnsembleModelTemplate:
    @pytest.fixture
    def dataset_table(self):
        dataset_table = MagicMock()
        dataset_table.return_value.__and__.return_value.__len__.return_value = 1
        dataset_table.return_value.proj.return_value.__and__.return_value.fetch1.return_value = dict(ds=0)
        return dataset_table

    @pytest.fixture
    def trained_model_table(self, model):
        trained_model_table = MagicMock()
        trained_model_table.return_value.proj.return_value.__and__.return_value.fetch.return_value = [
            dict(m=0),
            dict(m=1),
        ]
        trained_model_table.return_value.__and__.return_value.fetch.return_value = [dict(m=0, a=0), dict(m=1, a=1)]
        trained_model_table.return_value.load_model = MagicMock(
            side_effect=[("dataloaders1", model), ("dataloaders2", model)]
        )
        return trained_model_table

    @pytest.fixture
    def model(self):
        return MagicMock(side_effect=[torch.tensor([4.0, 7.0]), torch.tensor([6.0, 8.0])])

    @pytest.fixture
    def insert1(self):
        return MagicMock()

    @pytest.fixture
    def insert(self):
        return MagicMock()

    @pytest.fixture
    def trained_ensemble_model_template(self, dataset_table, trained_model_table, insert1, insert):
        trained_ensemble_model_template = tables.TrainedEnsembleModelTemplate
        trained_ensemble_model_template.dataset_table = dataset_table
        trained_ensemble_model_template.trained_model_table = trained_model_table
        trained_ensemble_model_template.insert1 = insert1
        trained_ensemble_model_template.Member.insert = insert
        return trained_ensemble_model_template

    @pytest.mark.parametrize(
        "n_datasets,expectation",
        [(0, pytest.raises(ValueError)), (1, does_not_raise()), (2, pytest.raises(ValueError))],
    )
    def test_if_key_correctness_is_checked(
        self, trained_ensemble_model_template, dataset_table, n_datasets, expectation
    ):
        dataset_table.return_value.__and__.return_value.__len__.return_value = n_datasets
        with expectation:
            trained_ensemble_model_template().create_ensemble("key")

    def test_if_dataset_key_is_correctly_fetched(self, trained_ensemble_model_template, dataset_table):
        trained_ensemble_model_template().create_ensemble("key")
        dataset_table.return_value.proj.return_value.__and__.assert_called_once_with("key")
        dataset_table.return_value.proj.return_value.__and__.return_value.fetch1.assert_called_once_with()

    def test_if_primary_model_keys_are_correctly_fetched(self, trained_ensemble_model_template, trained_model_table):
        trained_ensemble_model_template().create_ensemble("key")
        trained_model_table.return_value.proj.return_value.__and__.assert_called_once_with("key")
        trained_model_table.return_value.proj.return_value.__and__.return_value.fetch.assert_called_once_with(
            as_dict=True
        )

    def test_if_ensemble_key_is_correctly_inserted(self, trained_ensemble_model_template, insert1):
        trained_ensemble_model_template().create_ensemble("key")
        insert1.assert_called_once_with(dict(ds=0, ensemble_hash="536072017a2a3501ea8f09fffa51ee61"))

    def test_if_member_models_are_correctly_inserted(self, trained_ensemble_model_template, insert):
        trained_ensemble_model_template().create_ensemble("key")
        insert.assert_called_once_with(
            [
                dict(ds=0, ensemble_hash="536072017a2a3501ea8f09fffa51ee61", m=0),
                dict(ds=0, ensemble_hash="536072017a2a3501ea8f09fffa51ee61", m=1),
            ]
        )

    def test_if_model_keys_are_correctly_fetched(self, trained_ensemble_model_template, trained_model_table):
        trained_ensemble_model_template().load_model("key")
        trained_model_table.return_value.__and__.assert_called_once_with("key")
        trained_model_table.return_value.__and__.return_value.fetch.assert_called_once_with(as_dict=True)

    def test_if_models_are_correctly_loaded(self, trained_ensemble_model_template, trained_model_table):
        trained_ensemble_model_template().load_model("key")
        trained_model_table.return_value.load_model.assert_has_calls(
            [call(key=dict(m=0, a=0)), call(key=dict(m=1, a=1))]
        )

    def test_if_models_are_switched_to_eval_mode(self, trained_ensemble_model_template, model):
        trained_ensemble_model_template().load_model("key")
        model.eval.assert_has_calls([call(), call()])

    def test_if_ensemble_model_averaging_is_correct(self, trained_ensemble_model_template):
        _, model = trained_ensemble_model_template().load_model("key")
        assert torch.allclose(model("x"), torch.tensor([5.0, 7.5]))

    def test_if_only_first_dataloader_is_returned(self, trained_ensemble_model_template):
        dataloaders, _ = trained_ensemble_model_template().load_model("key")
        assert dataloaders == "dataloaders1"


class TestMEITemplate:
    @pytest.fixture
    def mei_template(self, trained_model_table, selector_table, method_table, insert1, save_func, model_loader_class):
        mei_template = tables.MEITemplate
        mei_template.trained_model_table = trained_model_table
        mei_template.selector_table = selector_table
        mei_template.method_table = method_table
        mei_template.insert1 = insert1
        mei_template.save_func = save_func
        mei_template.model_loader_class = model_loader_class
        temp_dir_func = MagicMock()
        temp_dir_func.return_value.__enter__.return_value = "/temp_dir"
        mei_template.temp_dir_func = temp_dir_func
        return mei_template

    @pytest.fixture
    def trained_model_table(self):
        return MagicMock()

    @pytest.fixture
    def selector_table(self):
        selector_table = MagicMock()
        selector_table.return_value.get_output_selected_model.return_value = "output_selected_model"
        return selector_table

    @pytest.fixture
    def method_table(self):
        method_table = MagicMock()
        mei_entity = MagicMock()
        mei_entity.squeeze.return_value = "mei"
        method_table.return_value.generate_mei.return_value = dict(mei=mei_entity)
        return method_table

    @pytest.fixture
    def insert1(self):
        return MagicMock()

    @pytest.fixture
    def save_func(self):
        return MagicMock()

    @pytest.fixture
    def model_loader_class(self, model_loader):
        return MagicMock(return_value=model_loader)

    @pytest.fixture
    def model_loader(self):
        model_loader = MagicMock()
        model_loader.load.return_value = "dataloaders", "model"
        return model_loader

    def test_if_model_loader_is_correctly_initialized(self, mei_template, trained_model_table, model_loader_class):
        mei_template(cache_size_limit=5)
        model_loader_class.assert_called_once_with(trained_model_table, cache_size_limit=5)

    def test_if_model_is_correctly_loaded(self, mei_template, model_loader):
        mei_template().make("key")
        model_loader.load.assert_called_once_with(key="key")

    def test_if_correct_model_output_is_selected(self, mei_template, selector_table):
        mei_template().make("key")
        selector_table.return_value.get_output_selected_model.assert_called_once_with("model", "key")

    def test_if_mei_is_correctly_generated(self, mei_template, method_table):
        mei_template().make("key")
        method_table.return_value.generate_mei.assert_called_once_with("dataloaders", "output_selected_model", "key")

    def test_if_mei_is_correctly_saved(self, mei_template, save_func):
        mei_template().make("key")
        save_func.assert_called_once_with("mei", "/temp_dir/d41d8cd98f00b204e9800998ecf8427e.pth.tar")

    def test_if_mei_entity_is_correctly_saved(self, mei_template, insert1):
        mei_template().make("key")
        insert1.assert_called_once_with(dict(mei="/temp_dir/d41d8cd98f00b204e9800998ecf8427e.pth.tar"))
