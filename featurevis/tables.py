from typing import Callable, Dict

import torch

from . import integration


class TrainedEnsembleModelTemplate:
    definition = """
    # contains ensemble ids
    -> self.dataset_table
    ensemble_hash : char(32) # the hash of the ensemble
    """

    class Member:
        definition = """
        # contains assignments of trained models to a specific ensemble id
        -> master
        -> master.trained_model_table
        """

        insert: Callable[[Dict], None]

    dataset_table = None
    trained_model_table = None

    insert1: Callable[[Dict], None]

    def create_ensemble(self, key):
        if len(self.dataset_table() & key) != 1:
            raise ValueError("Provided key not sufficient to restrict dataset table to one entry!")
        dataset_key = (self.dataset_table().proj() & key).fetch1()
        models = (self.trained_model_table().proj() & key).fetch(as_dict=True)
        ensemble_table_key = dict(dataset_key, ensemble_hash=integration.hash_list_of_dictionaries(models))
        self.insert1(ensemble_table_key)
        self.Member().insert([{**ensemble_table_key, **m} for m in models])

    def load_model(self, key=None):
        return self._load_ensemble_model(key=key)

    def _load_ensemble_model(self, key=None):
        def ensemble_model(x, *args, **kwargs):
            outputs = [m(x, *args, **kwargs) for m in models]
            mean_output = torch.stack(outputs, dim=0).mean(dim=0)
            return mean_output

        model_keys = (self.trained_model_table() & key).fetch(as_dict=True)
        dataloaders, models = tuple(
            list(x) for x in zip(*[self.trained_model_table().load_model(key=k) for k in model_keys])
        )
        for model in models:
            model.eval()
        return dataloaders[0], ensemble_model
