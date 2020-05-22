from unittest.mock import Mock, MagicMock

import pytest
import torch
from torch.nn import Module

from featurevis import integration


class FakeModel:
    def __init__(self, multiplier):
        self.multiplier = multiplier
        self.eval = Mock()

    def __call__(self, x, *args, **kwargs):
        if "data_key" in kwargs:
            x = x + kwargs["data_key"]
        return self.multiplier * x


class FakeMemberTable:
    # noinspection PyUnusedLocal
    @staticmethod
    def fetch(as_dict=False):
        return [dict(trained_model_attr=i) for i in range(3)]


@pytest.fixture
def fake_trained_model_table():
    def _fake_trained_model_table(primary_key=None):
        class FakeTrainedModelTable:
            primary_key = None
            models = []

            @classmethod
            def load_model(cls, key):
                model = FakeModel(key["trained_model_attr"] + 1)
                cls.models.append(model)
                return "dataloaders" + str(key["trained_model_attr"]), model

        setattr(FakeTrainedModelTable, "primary_key", primary_key)
        return FakeTrainedModelTable

    return _fake_trained_model_table


def get_fake_load_function(data):
    def fake_load_function(path):
        return data[path]

    return fake_load_function


def test_get_mappings():
    dataset_config = dict(datafiles=["path0", "path1"])
    key = dict(attr1=0)
    data = dict(
        path0=dict(unit_indices=["u0", "u1"], session_id="s0"), path1=dict(unit_indices=["u10"], session_id="s5")
    )
    mappings = integration.get_mappings(dataset_config, key, get_fake_load_function(data))
    assert mappings == [
        dict(attr1=0, neuron_id="u0", neuron_position=0, session_id="s0"),
        dict(attr1=0, neuron_id="u1", neuron_position=1, session_id="s0"),
        dict(attr1=0, neuron_id="u10", neuron_position=0, session_id="s5"),
    ]


def fake_get_dims(dataloaders):
    return dataloaders


class TestModelLoader:
    @pytest.mark.parametrize("order", ["same", "reversed"])
    def test_model_caching(self, fake_trained_model_table, order):
        key1 = dict(trained_model_attr=0, other_attr=1)
        if order == "same":
            key2 = key1.copy()
        else:
            key2 = {k: key1[k] for k in reversed(key1)}
        model_loader = integration.ModelLoader(fake_trained_model_table(primary_key=["trained_model_attr"]))
        model1 = model_loader.load(key1)
        model2 = model_loader.load(key2)
        assert model1 is model2

    @pytest.mark.parametrize("cache_size_limit", [0, 1, 10])
    def test_cache_size_limit(self, fake_trained_model_table, cache_size_limit):
        model_loader = integration.ModelLoader(
            fake_trained_model_table(primary_key=["trained_model_attr"]), cache_size_limit=cache_size_limit
        )
        first_model = model_loader.load(dict(trained_model_attr=0))
        for i in range(cache_size_limit):
            model_loader.load(dict(trained_model_attr=i + 1))
        model = model_loader.load(dict(trained_model_attr=0))
        assert model is not first_model


class TestHashListOfDictionaries:
    def test_output_format(self):
        list_of_dicts = [dict(a=1, b=2), dict(a=3, b=5), dict(a=2, b=8)]
        hashed = integration.hash_list_of_dictionaries(list_of_dicts)
        assert isinstance(hashed, str) and len(hashed) == 32

    def test_same_list_of_dicts(self):
        list_of_dicts = [dict(a=1, b=2), dict(a=3, b=5), dict(a=2, b=8)]
        assert self.hash_and_compare(list_of_dicts, list_of_dicts.copy())

    @staticmethod
    def hash_and_compare(list_of_dicts1, list_of_dicts2):
        hashed1, hashed2 = (integration.hash_list_of_dictionaries(x) for x in (list_of_dicts1, list_of_dicts2))
        return hashed1 == hashed2

    def test_invariance_to_dictionary_key_order(self):
        list_of_dicts1 = [dict(a=1, b=2), dict(a=3, b=5), dict(a=2, b=8)]
        list_of_dicts2 = [dict(a=1, b=2), dict(b=5, a=3), dict(a=2, b=8)]
        assert self.hash_and_compare(list_of_dicts1, list_of_dicts2)

    def test_invariance_to_order_of_dictionaries_in_list(self):
        list_of_dicts1 = [dict(a=3, b=5), dict(a=1, b=2), dict(a=2, b=8)]
        list_of_dicts2 = [dict(a=1, b=2), dict(a=3, b=5), dict(a=2, b=8)]
        assert self.hash_and_compare(list_of_dicts1, list_of_dicts2)


class TestEnsembleModel:
    @pytest.fixture
    def members(self):
        members = []
        for i in range(3):
            values = list(x + i * 3 for x in range(1, 4))
            member = MagicMock(name="member" + str(i + 1), return_value=torch.tensor([values], dtype=torch.float))
            member.__repr__ = MagicMock(return_value="member" + str(i + 1))
            members.append(member)
        return members

    def test_if_ensemble_model_is_pytorch_module(self):
        assert issubclass(integration.EnsembleModel, Module)

    def test_if_ensemble_model_initializes_super_class(self):
        class MockModule(Module):
            __init__ = MagicMock(name="__init__", return_value=None)

        class EnsembleModelTestable(integration.EnsembleModel, MockModule):
            pass

        EnsembleModelTestable()
        MockModule.__init__.assert_called_once_with()

    def test_if_input_is_passed_to_ensemble_members(self, members):
        ensemble = integration.EnsembleModel(*members)
        ensemble("x", "arg", kwarg="kwarg")
        for member in members:
            member.assert_called_once_with("x", "arg", kwarg="kwarg")

    def test_if_outputs_of_ensemble_members_is_correctly_averaged(self, members):
        ensemble = integration.EnsembleModel(*members)
        output = ensemble("x")
        assert torch.allclose(output, torch.tensor([4, 5, 6], dtype=torch.float))

    def test_repr(self, members):
        ensemble = integration.EnsembleModel(*members)
        assert str(ensemble) == "EnsembleModel(member1, member2, member3)"


class TestConstrainedOutputModel:
    @pytest.fixture
    def model(self):
        return MagicMock(return_value=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]))

    def test_if_constrained_output_model_is_pytorch_module(self):
        assert issubclass(integration.ConstrainedOutputModel, Module)

    def test_if_super_class_is_initialized(self, model):
        class MockModule(Module):
            __init__ = MagicMock(name="__init__", return_value=None)

        class ConstrainedOutputModelTestable(integration.ConstrainedOutputModel, MockModule):
            pass

        ConstrainedOutputModelTestable(model, 0)
        MockModule.__init__.assert_called_once_with()

    def test_if_input_is_passed_to_model(self, model):
        constrained_model = integration.ConstrainedOutputModel(model, 0)
        constrained_model("x", "arg", kwarg="kwarg")
        model.assert_called_once_with("x", "arg", kwarg="kwarg")

    @pytest.mark.parametrize("constraint,expected", [(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0), (4, 5.0)])
    def test_if_output_constraint_is_correct(self, model, constraint, expected):
        constrained_model = integration.ConstrainedOutputModel(model, constraint)
        output = constrained_model("x")
        assert torch.allclose(output, torch.tensor([expected]))

    def test_if_forward_kwargs_are_passed_to_model(self, model):
        constrained_model = integration.ConstrainedOutputModel(
            model, 0, forward_kwargs=dict(forward_kwarg="forward_kwarg")
        )
        constrained_model("x")
        model.assert_called_once_with("x", forward_kwarg="forward_kwarg")

    def test_repr(self):
        constrained_model = integration.ConstrainedOutputModel("model", 0, forward_kwargs=dict(kwarg="kwarg"))
        assert str(constrained_model) == "ConstrainedOutputModel(model, 0, forward_kwargs={'kwarg': 'kwarg'})"
