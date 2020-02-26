import tempfile
import os
import pickle

import torch

from nnfabrik.utility.nn_helpers import get_dims_for_loader_dict
from nnfabrik.utility.nnf_helper import split_module_name, dynamic_import
from nnfabrik.utility.dj_helpers import make_hash


class TrainedEnsembleModelHandler:
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

        def __init__(self, table):
            self.table = table

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
        ensemble_table_key = dict(dataset_key, ensemble_hash=hash_list_of_dictionaries(models))
        self.table.insert1(ensemble_table_key)
        self.table.Member().insert([{**ensemble_table_key, **m} for m in models])

    def load_model(self, key=None):
        """Wrapper to preserve the interface of the trained model table."""
        return load_ensemble_model(self.table.Member, self.table.trained_model_table, key=key)


class CSRFV1SelectorHandler:
    definition = """
    # contains information that can be used to map a neuron's id to its corresponding integer position in the output of
    # the model. 
    -> self.dataset_table
    neuron_id       : smallint unsigned # unique neuron identifier
    ---
    neuron_position : smallint unsigned # integer position of the neuron in the model's output 
    session_id      : varchar(13)       # unique session identifier
    """

    def __init__(self, table):
        self.table = table

    def make(self, key):
        dataset_config = (self.table.dataset_table & key).fetch1("dataset_config")
        mappings = get_mappings(dataset_config, key)
        self.table.insert(mappings)

    def get_output_selected_model(self, model, key):
        neuron_pos, session_id = (self.table & key).fetch1("neuron_position", "session_id")
        return get_output_selected_model(neuron_pos, session_id, model)


class MEIMethodHandler:
    definition = """
    # contains methods for generating MEIs and their configurations.
    method_fn                           : varchar(64)   # name of the method function
    method_hash                         : varchar(32)   # hash of the method config
    ---
    method_config                       : longblob      # method configuration object
    method_ts       = CURRENT_TIMESTAMP : timestamp     # UTZ timestamp at time of insertion
    """

    def __init__(self, table):
        self.table = table

    def add_method(self, method_fn, method_config):
        self.table.insert1(dict(method_fn=method_fn, method_hash=make_hash(method_config), method_config=method_config))

    def generate_mei(self, dataloader, model, key):
        method_fn, method_config = (self.table & key).fetch1("method_fn", "method_config")
        method_fn = import_module(method_fn)
        mei, evaluations = method_fn(dataloader, model, method_config)
        return dict(key, evaluations=evaluations, mei=mei)


class MEIHandler:
    definition = """
    # contains maximally exciting images (MEIs)
    -> self.method_table
    -> self.trained_model_table
    -> self.selector_table
    ---
    mei                 : attach@minio  # the MEI as a tensor
    evaluations         : longblob      # list of function evaluations at each iteration in the mei generation process 
    """

    def __init__(self, table, cache_size_limit=10):
        self.table = table
        self.model_loader = ModelLoader(table.trained_model_table, cache_size_limit=cache_size_limit)

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


def load_ensemble_model(member_table, trained_model_table, key=None):
    """Loads an ensemble model.

    Args:
        member_table: A Datajoint table containing a subset of the trained models in the trained model table.
        trained_model_table: A Datajoint table containing trained models. Must have a method called "load_model" which
            must itself return a PyTorch module.
        key: A dictionary used to restrict the member table.

    Returns:
        A function that has the model's input as parameters and returns the mean output across the individual models
        in the ensemble.
    """

    def ensemble_model(x, *args, **kwargs):
        outputs = [m(x, *args, **kwargs) for m in models]
        mean_output = torch.stack(outputs, dim=0).mean(dim=0)
        return mean_output

    if key:
        query = member_table() & key
    else:
        query = member_table()
    model_keys = query.fetch(as_dict=True)
    dataloaders, models = tuple(list(x) for x in zip(*[trained_model_table().load_model(key=k) for k in model_keys]))
    for model in models:
        model.eval()
    return dataloaders[0], ensemble_model


def get_output_selected_model(neuron_pos, session_id, model):
    """Creates a version of the model that has its output selected down to a single uniquely identified neuron.

    Args:
        neuron_pos: An integer, the position of the neuron in the model's output.
        session_id: A string that uniquely identifies one of the model's readouts.
        model: A PyTorch module that can be called with a keyword argument called "data_key". The output of the
            module is expected to be a two dimensional Torch tensor where the first dimension corresponds to the
            batch size and the second to the number of neurons.

    Returns:
        A function that takes the model input(s) as parameter(s) and returns the model output corresponding to the
        selected neuron.
    """

    def output_selected_model(x, *args, **kwargs):
        output = model(x, *args, data_key=session_id, **kwargs)
        return output[:, neuron_pos]

    return output_selected_model


def get_mappings(dataset_config, key, load_func=None):
    if load_func is None:
        load_func = load_pickled_data
    entities = []
    for datafile_path in dataset_config["datafiles"]:
        data = load_func(datafile_path)
        for neuron_pos, neuron_id in enumerate(data["unit_indices"]):
            entities.append(dict(key, neuron_id=neuron_id, neuron_position=neuron_pos, session_id=data["session_id"]))
    return entities


def load_pickled_data(path):
    with open(path, "rb") as datafile:
        data = pickle.load(datafile)
    return data


def get_input_shape(dataloaders, get_dims_func=get_dims_for_loader_dict):
    """Gets the shape of the input that the model expects from the dataloaders."""
    return list(get_dims_func(dataloaders["train"]).values())[0]["inputs"]


def import_module(path):
    return dynamic_import(*split_module_name(path))


class ModelLoader:
    def __init__(self, model_table, cache_size_limit=10):
        self.model_table = model_table
        self.cache_size_limit = cache_size_limit
        self.cache = dict()

    def load(self, key):
        if self.cache_size_limit == 0:
            return self._load_model(key)
        if not self._is_cached(key):
            self._cache_model(key)
        return self._get_cached_model(key)

    def _load_model(self, key):
        return self.model_table().load_model(key=key)

    def _is_cached(self, key):
        if self._hash_trained_model_key(key) in self.cache:
            return True
        return False

    def _cache_model(self, key):
        """Caches a model and makes sure the cache is not bigger than the specified limit."""
        self.cache[self._hash_trained_model_key(key)] = self._load_model(key)
        if len(self.cache) > self.cache_size_limit:
            del self.cache[list(self.cache)[0]]

    def _get_cached_model(self, key):
        return self.cache[self._hash_trained_model_key(key)]

    def _hash_trained_model_key(self, key):
        """Creates a hash from the part of the key corresponding to the primary key of the trained model table."""
        return make_hash({k: key[k] for k in self.model_table().primary_key})


def hash_list_of_dictionaries(list_of_dicts):
    """Creates a hash from a list of dictionaries that uniquely identifies the provided list of dictionaries.

    The keys of every dictionary in the list and the list itself are sorted before creating the hash.

    Args:
        list_of_dicts: List of dictionaries.

    Returns:
        A string representing the hash that uniquely identifies the provided list of dictionaries.
    """
    dict_of_dicts = {make_hash(d): d for d in list_of_dicts}
    sorted_list_of_dicts = [dict_of_dicts[h] for h in sorted(dict_of_dicts)]
    return make_hash(sorted_list_of_dicts)
