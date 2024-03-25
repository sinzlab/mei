"""This module contains classes and functions pertaining to the NNFabrik integration."""

import pickle
from copy import deepcopy

from nnfabrik.utility.dj_helpers import make_hash
from nnfabrik.utility.nnf_helper import dynamic_import, split_module_name


def load_pickled_data(path):
    with open(path, "rb") as datafile:
        data = pickle.load(datafile)
    return data


def get_mappings(dataset_config, key, load_func=load_pickled_data):
    entities = []
    for datafile_path in dataset_config["datafiles"]:
        data = load_func(datafile_path)
        for neuron_pos, neuron_id in enumerate(data["unit_indices"]):
            entities.append(
                dict(
                    key,
                    neuron_id=neuron_id,
                    neuron_position=neuron_pos,
                    session_id=data["session_id"],
                )
            )
    return entities


def import_module(path):
    return dynamic_import(*split_module_name(path))


class ModelLoader:
    """
    A utility class for loading and caching models.

    This class is designed to manage the loading and caching of models to optimize performance by reducing redundant
    loading operations. It supports limiting the cache size to prevent excessive memory usage.

    Attributes:
        model_table (callable): A callable that returns an object capable of loading models given a key.
        cache_size_limit (int, optional): The maximum number of models to keep in the cache. Defaults to 10.
        cache (dict): A dictionary acting as the cache for storing loaded models.

    Methods:
        load(key): Loads a model by its key, using the cache if available. Loading it from the `model_table` otherwise.
        _load_model(key): Directly loads a model from the `model_table` without attempting to use the cache.
        _is_cached(key): Checks if a model corresponding to the given key is present in the cache.
        _cache_model(key): Caches a model under the given key, ensuring the cache does not exceed its size limit.
        _get_cached_model(key): Retrieves a model from the cache based on its key.
        _hash_trained_model_key(key): Generates a hash for the model key, primarily for use as a cache key.

    The load method is the primary interface for users of this class, automatically managing caching to optimize
    model loading. The cache is transparently managed according to the specified cache size limit.
    """

    def __init__(self, model_table, cache_size_limit=10):
        self.model_table = model_table
        self.cache_size_limit = cache_size_limit
        self.cache = dict()

    def load(self, key, **kwargs):
        if self.cache_size_limit == 0:
            return self._load_model(key, **kwargs)
        if not self._is_cached(key):
            self._cache_model(key, **kwargs)
        return deepcopy(self._get_cached_model(key))

    def _load_model(self, key, **kwargs):
        return self.model_table().load_model(key=key, **kwargs)

    def _is_cached(self, key):
        if self._hash_trained_model_key(key) in self.cache:
            return True
        return False

    def _cache_model(self, key, **kwargs):
        """Caches a model and makes sure the cache is not bigger than the specified limit."""
        self.cache[self._hash_trained_model_key(key)] = self._load_model(key, **kwargs)
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
