import pytest
import torch

from featurevis import integration


class FakeModel:
    def __init__(self, multiplier):
        self.multiplier = multiplier

    def __call__(self, x, *args, **kwargs):
        if "data_key" in kwargs:
            x = x + kwargs["data_key"]
        return self.multiplier * x


class FakeMemberTable:
    # noinspection PyUnusedLocal
    @staticmethod
    def fetch(as_dict=False):
        return [dict(trained_model_attr=i) for i in range(3)]


def get_fake_trained_model_table(primary_key=None):
    class FakeTrainedModelTable:
        primary_key = None
        # noinspection PyUnusedLocal
        @staticmethod
        def load_model(key):
            return "dataloaders" + str(key["trained_model_attr"]), FakeModel(key["trained_model_attr"] + 1)

    setattr(FakeTrainedModelTable, "primary_key", primary_key)
    return FakeTrainedModelTable


def test_load_ensemble():
    dataloaders, ensemble_model = integration.load_ensemble_model(FakeMemberTable, get_fake_trained_model_table())
    ensemble_input = torch.tensor([1, 2, 3], dtype=torch.float)
    expected_output = torch.tensor([2, 4, 6], dtype=torch.float)
    assert dataloaders == "dataloaders0"
    assert torch.allclose(ensemble_model(ensemble_input), expected_output)


def test_get_output_selected_model():
    model = integration.get_output_selected_model(0, 10, FakeModel(1))
    output = model(torch.tensor([[1, 2, 3]], dtype=torch.float))
    expected_output = torch.tensor([[11]], dtype=torch.float)
    assert output == expected_output


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


def test_get_input_shape():
    dataloaders = dict(
        train=dict(session_id0=dict(inputs=0), session_id1=dict(inputs=1)),
        validation=dict(session_id0=dict(inputs=2), session_id1=dict(inputs=3)),
    )
    shape = integration.get_input_shape(dataloaders, get_dims_func=fake_get_dims)
    assert shape == 0


@pytest.mark.parametrize("raw_optim_kwargs,optim_kwargs", [(None, dict()), (dict(a=1), dict(a=1))])
def test_prepare_mei_method(raw_optim_kwargs, optim_kwargs):
    method = dict(
        method_id=0,
        optim_kwargs=raw_optim_kwargs,
        transform="module0.func1",
        regularization=None,
        gradient_f="module3.func6",
        post_update=None,
    )
    prepared = integration.prepare_mei_method(method, import_func=lambda x: x)
    expected = dict(
        optim_kwargs=optim_kwargs,
        transform="module0.func1",
        regularization=None,
        gradient_f="module3.func6",
        post_update=None,
    )
    assert prepared == expected


class TestModelLoader:
    @pytest.mark.parametrize("order", ["same", "reversed"])
    def test_model_caching(self, order):
        key1 = dict(trained_model_attr=0, other_attr=1)
        if order == "same":
            key2 = key1.copy()
        else:
            key2 = {k: key1[k] for k in reversed(key1)}
        model_loader = integration.ModelLoader(get_fake_trained_model_table(primary_key=["trained_model_attr"]))
        model1 = model_loader.load(key1)
        model2 = model_loader.load(key2)
        assert model1 is model2

    @pytest.mark.parametrize("cache_size_limit", [0, 1, 10])
    def test_cache_size_limit(self, cache_size_limit):
        model_loader = integration.ModelLoader(
            get_fake_trained_model_table(primary_key=["trained_model_attr"]), cache_size_limit=cache_size_limit
        )
        first_model = model_loader.load(dict(trained_model_attr=0))
        for i in range(cache_size_limit):
            model_loader.load(dict(trained_model_attr=i + 1))
        model = model_loader.load(dict(trained_model_attr=0))
        assert model is not first_model
