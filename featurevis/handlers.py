import tempfile
import os

import torch

from nnfabrik.utility.dj_helpers import make_hash
from . import integration


class TrainedEnsembleModelHandler:
    def __init__(self, table):
        self.table = table

    def create_ensemble(self, key):
        """Creates a new ensemble and inserts it into the table.

        Args:
            key: A dictionary representing a key that must be sufficient to restrict the dataset table to one entry. The
                models that are in the trained model table after restricting it with the provided key will be part of
                the ensemble.

        Returns:
            None.
        """
        if len(self.table.dataset_table() & key) != 1:
            raise ValueError("Provided key not sufficient to restrict dataset table to one entry!")
        dataset_key = (self.table.dataset_table().proj() & key).fetch1()
        models = (self.table.trained_model_table().proj() & key).fetch(as_dict=True)
        ensemble_table_key = dict(dataset_key, ensemble_hash=integration.hash_list_of_dictionaries(models))
        self.table.insert1(ensemble_table_key)
        self.table.Member().insert([{**ensemble_table_key, **m} for m in models])

    def load_model(self, key=None):
        """Wrapper to preserve the interface of the trained model table."""
        return integration.load_ensemble_model(self.table.Member, self.table.trained_model_table, key=key)


class CSRFV1SelectorHandler:
    def __init__(self, table):
        self.table = table

    def make(self, key):
        dataset_config = (self.table.dataset_table & key).fetch1("dataset_config")
        mappings = integration.get_mappings(dataset_config, key)
        self.table.insert(mappings)

    def get_output_selected_model(self, model, key):
        neuron_pos, session_id = (self.table & key).fetch1("neuron_position", "session_id")
        return integration.get_output_selected_model(neuron_pos, session_id, model)


class MEIMethodHandler:
    def __init__(self, table):
        self.table = table

    def add_method(self, method_fn, method_config):
        self.table.insert1(dict(method_fn=method_fn, method_hash=make_hash(method_config), method_config=method_config))

    def generate_mei(self, dataloader, model, key):
        method_fn, method_config = (self.table & key).fetch1("method_fn", "method_config")
        method_fn = integration.import_module(method_fn)
        mei, evaluations = method_fn(dataloader, model, method_config)
        return dict(key, evaluations=evaluations, mei=mei)


class MEIHandler:
    def __init__(self, table, cache_size_limit=10):
        self.table = table
        self.model_loader = integration.ModelLoader(table.trained_model_table, cache_size_limit=cache_size_limit)

    def make(self, key):
        dataloaders, model = self.model_loader.load(key=key)
        output_selected_model = self.table.selector_table().get_output_selected_model(model, key)
        mei_entity = self.table.method_table().generate_mei(dataloaders, output_selected_model, key)
        self._insert_mei(self.table, mei_entity)

    @staticmethod
    def _insert_mei(table, mei_entity):
        """Saves the MEI to a temporary directory and inserts the prepared entity into the table."""
        mei = mei_entity.pop("mei").squeeze()
        filename = make_hash(mei_entity) + ".pth.tar"
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, filename)
            torch.save(mei, filepath)
            mei_entity["mei"] = filepath
            table.insert1(mei_entity)
