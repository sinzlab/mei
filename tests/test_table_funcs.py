import torch

from featurevis import table_funcs


class FakeModel:
    def __init__(self, multiplier):
        self.multiplier = multiplier

    def __call__(self, x, *args, **kwargs):
        if 'data_key' in kwargs:
            x = x + kwargs['data_key']
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


class FakeCSRFV1Selector:
    def __and__(self, other):
        return self

    @staticmethod
    def fetch1(*_args):
        return 0, 10


def test_get_output_selected_model():
    model = table_funcs.get_output_selected_model(FakeCSRFV1Selector(), FakeModel(1), 'dummy_key')
    output = model(torch.tensor([[1, 2, 3]], dtype=torch.float))
    expected_output = torch.tensor([[11]], dtype=torch.float)
    assert output == expected_output
