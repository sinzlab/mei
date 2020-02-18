import torch

from featurevis import table_funcs


class FakeModel:
    def __init__(self, multiplier):
        self.multiplier = multiplier

    def __call__(self, x, *args, **kwargs):
        if "data_key" in kwargs:
            x = x + kwargs["data_key"]
        return self.multiplier * x


class FakeMember:
    # noinspection PyUnusedLocal
    @staticmethod
    def fetch(as_dict=False):
        return ["key1", "key2", "key3"]


class FakeTrainedModel:
    # noinspection PyUnusedLocal
    @staticmethod
    def load_model(key):
        return "dataloader" + key[-1], FakeModel(int(key[-1]))


def test_load_ensemble():
    dataloader, ensemble_model = table_funcs.load_ensemble_model(FakeMember, FakeTrainedModel)
    ensemble_input = torch.tensor([1, 2, 3], dtype=torch.float)
    expected_output = torch.tensor([2, 4, 6], dtype=torch.float)
    assert dataloader == "dataloader1"
    assert torch.allclose(ensemble_model(ensemble_input), expected_output)


def test_get_output_selected_model():
    model = table_funcs.get_output_selected_model(0, 10, FakeModel(1))
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
    mappings = table_funcs.get_mappings(dataset_config, key, get_fake_load_function(data))
    assert mappings == [
        dict(attr1=0, neuron_id="u0", neuron_position=0, session_id="s0"),
        dict(attr1=0, neuron_id="u1", neuron_position=1, session_id="s0"),
        dict(attr1=0, neuron_id="u10", neuron_position=0, session_id="s5"),
    ]
