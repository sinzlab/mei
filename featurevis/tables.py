import os
import tempfile
from typing import Callable, Dict

import torch
from nnfabrik.utility.dj_helpers import make_hash

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


class MEITemplate:
    definition = """
    # contains maximally exciting images (MEIs)
    -> self.method_table
    -> self.trained_model_table
    -> self.selector_table
    ---
    mei                 : attach@minio  # the MEI as a tensor
    evaluations         : longblob      # list of function evaluations at each iteration in the mei generation process 
    """

    trained_model_table = None
    selector_table = None
    method_table = None
    model_loader_class = integration.ModelLoader
    save_func = staticmethod(torch.save)
    temp_dir_func = tempfile.TemporaryDirectory

    insert1: Callable[[Dict], None]

    def __init__(self, *args, cache_size_limit=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_loader = self.model_loader_class(self.trained_model_table, cache_size_limit=cache_size_limit)

    def make(self, key):
        dataloaders, model = self.model_loader.load(key=key)
        output_selected_model = self.selector_table().get_output_selected_model(model, key)
        mei_entity = self.method_table().generate_mei(dataloaders, output_selected_model, key)
        self._insert_mei(mei_entity)

    def _insert_mei(self, mei_entity):
        """Saves the MEI to a temporary directory and inserts the prepared entity into the table."""
        mei = mei_entity.pop("mei").squeeze()
        filename = make_hash(mei_entity) + ".pth.tar"
        with self.temp_dir_func() as temp_dir:
            filepath = os.path.join(temp_dir, filename)
            self.save_func(mei, filepath)
            mei_entity["mei"] = filepath
            self.insert1(mei_entity)
