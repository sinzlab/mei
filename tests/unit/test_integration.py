from unittest.mock import Mock

import pytest

from mei import integration


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
