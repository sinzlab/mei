from unittest.mock import MagicMock, call
from contextlib import contextmanager

import pytest

from featurevis import mixins


@pytest.fixture
def key():
    return MagicMock(name="key")


@contextmanager
def does_not_raise():
    yield


@pytest.fixture
def trained_ensemble_model_template(
    member_table, dataset_table, trained_model_table, ensemble_model_class, insert1, fetch1, magic_and
):
    trained_ensemble_model_template = mixins.TrainedEnsembleModelTemplateMixin
    trained_ensemble_model_template.Member = member_table
    trained_ensemble_model_template.dataset_table = dataset_table
    trained_ensemble_model_template.trained_model_table = trained_model_table
    trained_ensemble_model_template.ensemble_model_class = ensemble_model_class
    trained_ensemble_model_template.insert1 = insert1
    trained_ensemble_model_template.fetch1 = fetch1
    trained_ensemble_model_template.__and__ = magic_and
    return trained_ensemble_model_template


@pytest.fixture
def member_table():
    member_table = MagicMock(name="member_table", spec=mixins.TrainedEnsembleModelTemplateMixin.Member)
    member_table.return_value.__and__.return_value.fetch = MagicMock(
        name="member_and", return_value=[dict(m=0, a=0), dict(m=1, a=1)]
    )
    return member_table


@pytest.fixture
def dataset_table():
    dataset_table = MagicMock()
    dataset_table.return_value.__and__.return_value.__len__.return_value = 1
    dataset_table.return_value.proj.return_value.__and__.return_value.fetch1.return_value = dict(ds=0)
    return dataset_table


@pytest.fixture
def trained_model_table():
    trained_model_table = MagicMock()
    trained_model_table.return_value.proj.return_value.__and__.return_value.fetch.return_value = [
        dict(m=0),
        dict(m=1),
    ]
    trained_model_table.return_value.load_model = MagicMock(
        side_effect=[("dataloaders1", "model1"), ("dataloaders2", "model2")]
    )
    return trained_model_table


@pytest.fixture()
def ensemble_model_class():
    return MagicMock()


@pytest.fixture
def insert1():
    return MagicMock()


@pytest.fixture
def fetch1():
    return MagicMock(name="fetch1", return_value="key")


@pytest.fixture
def magic_and():
    magic_and = MagicMock(name="and")
    magic_and.return_value.fetch1 = MagicMock(name="fetch1", return_value="ensemble_model_key")
    return magic_and


class TestCreateEnsemble:
    @pytest.mark.parametrize(
        "n_datasets,expectation",
        [(0, pytest.raises(ValueError)), (1, does_not_raise()), (2, pytest.raises(ValueError))],
    )
    def test_if_key_correctness_is_checked(
        self, key, trained_ensemble_model_template, dataset_table, n_datasets, expectation
    ):
        dataset_table.return_value.__and__.return_value.__len__.return_value = n_datasets
        with expectation:
            trained_ensemble_model_template().create_ensemble(key)

    def test_if_dataset_key_is_correctly_fetched(self, key, trained_ensemble_model_template, dataset_table):
        trained_ensemble_model_template().create_ensemble(key)
        dataset_table.return_value.proj.return_value.__and__.assert_called_once_with(key)
        dataset_table.return_value.proj.return_value.__and__.return_value.fetch1.assert_called_once_with()

    def test_if_primary_model_keys_are_correctly_fetched(
        self, key, trained_ensemble_model_template, trained_model_table
    ):
        trained_ensemble_model_template().create_ensemble(key)
        trained_model_table.return_value.proj.return_value.__and__.assert_called_once_with(key)
        trained_model_table.return_value.proj.return_value.__and__.return_value.fetch.assert_called_once_with(
            as_dict=True
        )

    def test_if_ensemble_key_is_correctly_inserted(self, key, trained_ensemble_model_template, insert1):
        trained_ensemble_model_template().create_ensemble(key)
        insert1.assert_called_once_with(
            dict(ds=0, ensemble_hash="536072017a2a3501ea8f09fffa51ee61", ensemble_comment="")
        )

    def test_if_member_models_are_correctly_inserted(self, key, trained_ensemble_model_template, member_table):
        trained_ensemble_model_template().create_ensemble(key)
        member_table.return_value.insert.assert_called_once_with(
            [
                dict(ds=0, ensemble_hash="536072017a2a3501ea8f09fffa51ee61", m=0),
                dict(ds=0, ensemble_hash="536072017a2a3501ea8f09fffa51ee61", m=1),
            ]
        )


class TestLoadModel:
    def test_if_key_is_fetched_if_not_provided(self, key, trained_ensemble_model_template, fetch1, magic_and):
        trained_ensemble_model_template().load_model()
        fetch1.assert_called_once_with("KEY")
        magic_and.assert_called_once_with("key")

    def test_if_model_keys_are_correctly_fetched(self, key, trained_ensemble_model_template, magic_and, member_table):
        trained_ensemble_model_template().load_model(key)
        magic_and.assert_called_once_with(key)
        magic_and.return_value.fetch1.assert_called_once_with()
        member_table.return_value.__and__.assert_called_once_with("ensemble_model_key")
        member_table.return_value.__and__.return_value.fetch.assert_called_once_with(as_dict=True)

    def test_if_models_are_correctly_loaded(self, key, trained_ensemble_model_template, trained_model_table):
        trained_ensemble_model_template().load_model(key)
        trained_model_table.return_value.load_model.assert_has_calls(
            [call(key=dict(m=0, a=0)), call(key=dict(m=1, a=1))]
        )

    def test_if_ensemble_model_is_correctly_initialized(
        self, key, trained_ensemble_model_template, ensemble_model_class
    ):
        trained_ensemble_model_template().load_model(key)
        ensemble_model_class.assert_called_once_with("model1", "model2")

    def test_if_only_first_dataloader_is_returned(self, key, trained_ensemble_model_template):
        dataloaders, _ = trained_ensemble_model_template().load_model(key)
        assert dataloaders == "dataloaders1"
