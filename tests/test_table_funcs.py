import torch

from featurevis import table_funcs


class FakeModel:
    @staticmethod
    def __call__(x, *args, **kwargs):
        return x


class FakeMember:
    # noinspection PyUnusedLocal
    @staticmethod
    def fetch(as_dict=False):
        return ["key1", "key2", "key3"]


class FakeTrainedModel:
    # noinspection PyUnusedLocal
    @staticmethod
    def load_model(key=None):
        return "dataloaders", FakeModel()


def test_load_ensemble():
    dataloader, ensemble_model = table_funcs.load_ensemble_model(FakeMember, FakeTrainedModel)
    ensemble_input = torch.tensor([1, 2, 3], dtype=torch.float)
    assert dataloader == "dataloaders"
    assert torch.allclose(ensemble_model(ensemble_input), ensemble_input)
