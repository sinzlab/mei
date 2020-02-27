from unittest.mock import MagicMock

import pytest

from featurevis import facades


class TestTrainedEnsembleModelFacade:
    def test_that_call_to_dataset_table_in_properly_restricts_is_correct(self):
        projected_dataset_table_instance = MagicMock()
        dataset_table_instance = MagicMock()
        dataset_table_instance.proj.return_value = projected_dataset_table_instance
        dataset_table = MagicMock(return_value=dataset_table_instance)

        facade = facades.TrainedEnsembleModelFacade(None, None, dataset_table, None)
        facade.properly_restricts("key")

        dataset_table.assert_called_once_with()
        dataset_table_instance.proj.assert_called_once_with()
        projected_dataset_table_instance.__and__.assert_called_once_with("key")

    @pytest.mark.parametrize("length,expected", [(0, False), (1, True), (2, False)])
    def test_properly_restricts(self, length, expected):
        restricted_projected_dataset_table_instance = MagicMock()
        restricted_projected_dataset_table_instance.__len__.return_value = length
        projected_dataset_table_instance = MagicMock()
        projected_dataset_table_instance.__and__.return_value = restricted_projected_dataset_table_instance
        dataset_table_instance = MagicMock()
        dataset_table_instance.proj.return_value = projected_dataset_table_instance
        dataset_table = MagicMock(return_value=dataset_table_instance)

        facade = facades.TrainedEnsembleModelFacade(None, None, dataset_table, None)
        is_properly_restricted = facade.properly_restricts("key")

        assert is_properly_restricted is expected

    def test_that_call_to_dataset_table_in_fetch_primary_dataset_key_is_correct(self):
        restricted_projected_dataset_table_instance = MagicMock()
        projected_dataset_table_instance = MagicMock()
        projected_dataset_table_instance.__and__.return_value = restricted_projected_dataset_table_instance
        dataset_table_instance = MagicMock()
        dataset_table_instance.proj.return_value = projected_dataset_table_instance
        dataset_table = MagicMock(return_value=dataset_table_instance)

        facade = facades.TrainedEnsembleModelFacade(None, None, dataset_table, None)
        facade.fetch_primary_dataset_key("key")

        dataset_table.assert_called_once_with()
        dataset_table_instance.proj.assert_called_once_with()
        projected_dataset_table_instance.__and__.assert_called_once_with("key")
        restricted_projected_dataset_table_instance.fetch1.assert_called_once_with()

    @pytest.mark.parametrize("key,expected", [("key", "key"), (None, dict())])
    def test_that_call_to_trained_model_table_in_fetch_trained_models_is_correct(self, key, expected):
        restricted_trained_model_table_instance = MagicMock()
        trained_model_table_instance = MagicMock()
        trained_model_table_instance.__and__.return_value = restricted_trained_model_table_instance
        trained_model_table = MagicMock(return_value=trained_model_table_instance)

        facade = facades.TrainedEnsembleModelFacade(None, None, None, trained_model_table)
        facade.fetch_trained_models(key)

        trained_model_table.assert_called_once_with()
        trained_model_table_instance.__and__.assert_called_once_with(expected)
        restricted_trained_model_table_instance.fetch.assert_called_once_with(as_dict=True)

    def test_that_call_to_trained_model_table_in_fetch_trained_model_primary_keys_is_correct(self):
        restricted_projected_trained_model_table_instance = MagicMock()
        projected_trained_model_table_instance = MagicMock()
        projected_trained_model_table_instance.__and__.return_value = restricted_projected_trained_model_table_instance
        trained_model_table_instance = MagicMock()
        trained_model_table_instance.proj.return_value = projected_trained_model_table_instance
        trained_model_table = MagicMock(return_value=trained_model_table_instance)

        facade = facades.TrainedEnsembleModelFacade(None, None, None, trained_model_table)
        facade.fetch_trained_models_primary_keys("key")

        trained_model_table.assert_called_once_with()
        trained_model_table_instance.proj.assert_called_once_with()
        projected_trained_model_table_instance.__and__.assert_called_once_with("key")
        restricted_projected_trained_model_table_instance.fetch.assert_called_once_with(as_dict=True)

    def test_that_call_to_load_model_is_correct(self):
        trained_model_table_instance = MagicMock()
        trained_model_table = MagicMock(return_value=trained_model_table_instance)

        facade = facades.TrainedEnsembleModelFacade(None, None, None, trained_model_table)
        facade.load_model("key")

        trained_model_table_instance.load_model.assert_called_once_with(key="key")

    def test_that_call_to_insert1_is_correct(self):
        trained_ensemble_table_instance = MagicMock()
        trained_ensemble_table = MagicMock(return_value=trained_ensemble_table_instance)

        facade = facades.TrainedEnsembleModelFacade(trained_ensemble_table, None, None, None)
        facade.insert_ensemble("key")

        trained_ensemble_table_instance.insert1.assert_called_once_with("key")

    def test_that_call_to_insert_is_correct(self):
        member_table_instance = MagicMock()
        member_table = MagicMock(return_value=member_table_instance)

        facade = facades.TrainedEnsembleModelFacade(None, member_table, None, None)
        facade.insert_members("members")

        member_table_instance.insert.assert_called_once_with("members")
