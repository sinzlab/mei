from unittest.mock import MagicMock, call
from contextlib import contextmanager

import pytest
import torch

from featurevis import handlers


@contextmanager
def does_not_raise():
    yield


class TestTrainedEnsembleModel:
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
            handler.create_ensemble(None)

    def test_if_call_to_properly_restricts_is_correct(self, facade, handler):
        handler.create_ensemble("key")

        facade.properly_restricts.assert_called_once_with("key")

    def test_if_call_to_fetch_trained_models_primary_keys_is_correct(self, facade, handler):
        handler.create_ensemble("key")

        facade.fetch_trained_models_primary_keys.assert_called_once_with("key")

    def test_if_call_to_insert_ensemble_is_correct(self, facade, handler):
        facade.fetch_primary_dataset_key = MagicMock(return_value=dict(d1=1, d2=2))

        handler.create_ensemble(None)

        facade.insert_ensemble.assert_called_once_with(
            dict(d1=1, d2=2, ensemble_hash="d41d8cd98f00b204e9800998ecf8427e")
        )

    def test_if_call_to_insert_members_is_correct(self, facade, handler):
        facade.fetch_primary_dataset_key = MagicMock(return_value=dict(d=1))
        facade.fetch_trained_models_primary_keys = MagicMock(return_value=[dict(m=0), dict(m=1)])

        handler.create_ensemble(None)

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

        assert torch.allclose(ensemble(None), torch.tensor([5, 7.5]))

    def test_that_only_first_dataloader_is_returned(self, facade, handler):
        facade.fetch_trained_models = MagicMock(return_value=["m1", "m2"])
        dataloaders = [("dataloader1", MagicMock()), ("dataloader2", MagicMock())]
        facade.load_model = MagicMock(side_effect=dataloaders)

        dataloader, _ = handler.load_model("key")

        assert dataloader == "dataloader1"
