"""This module contains mix-ins for the main tables and table templates."""

from __future__ import annotations
import os
import tempfile
from typing import Callable, Iterable, Mapping, Optional, Tuple, Dict, Any
from string import ascii_letters
from random import choice

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from nnfabrik.utility.dj_helpers import make_hash

from . import integration
from .modules import EnsembleModel, ConstrainedOutputModel


Key = Dict[str, Any]
Dataloaders = Dict[str, DataLoader]


class TrainedEnsembleModelTemplateMixin:
    definition = """
    # contains ensemble ids
    -> self.dataset_table
    ensemble_hash                   : char(32)      # the hash of the ensemble
    ---
    ensemble_comment        = ''    : varchar(256)  # a short comment describing the ensemble
    """

    class Member:
        definition = """
        # contains assignments of trained models to a specific ensemble id
        -> master
        -> master.trained_model_table
        """

        insert: Callable[[Iterable], None]
        __and__: Callable[[Key], TrainedEnsembleModelTemplateMixin.Member]
        fetch: Callable

    dataset_table = None
    trained_model_table = None
    ensemble_model_class = EnsembleModel

    insert1: Callable[[Mapping], None]
    __and__: Callable[[Key], TrainedEnsembleModelTemplateMixin]
    fetch1: Callable

    def create_ensemble(self, key: Key, comment: str = "") -> None:
        if len(self.dataset_table() & key) != 1:
            raise ValueError("Provided key not sufficient to restrict dataset table to one entry!")
        dataset_key = (self.dataset_table().proj() & key).fetch1()
        models = (self.trained_model_table().proj() & key).fetch(as_dict=True)
        primary_key = dict(dataset_key, ensemble_hash=integration.hash_list_of_dictionaries(models))
        self.insert1(dict(primary_key, ensemble_comment=comment))
        self.Member().insert([{**primary_key, **m} for m in models])

    def load_model(self, key: Optional[Key] = None) -> Tuple[Dataloaders, EnsembleModel]:
        if key is None:
            key = self.fetch1("KEY")
        return self._load_ensemble_model(key=key)

    def _load_ensemble_model(self, key: Optional[Key] = None) -> Tuple[Dataloaders, EnsembleModel]:
        ensemble_key = (self & key).fetch1()
        model_keys = (self.Member() & ensemble_key).fetch(as_dict=True)
        dataloaders, models = tuple(
            list(x) for x in zip(*[self.trained_model_table().load_model(key=k) for k in model_keys])
        )
        return dataloaders[0], self.ensemble_model_class(*models)


class CSRFV1SelectorTemplateMixin:
    definition = """
    # contains information that can be used to map a neuron's id to its corresponding integer position in the output of
    # the model. 
    -> self.dataset_table
    neuron_id       : smallint unsigned # unique neuron identifier
    ---
    neuron_position : smallint unsigned # integer position of the neuron in the model's output 
    session_id      : varchar(13)       # unique session identifier
    """

    dataset_table = None
    dataset_fn = "csrf_v1"
    constrained_output_model = ConstrainedOutputModel

    insert: Callable[[Iterable], None]
    __and__: Callable[[Mapping], CSRFV1SelectorTemplateMixin]
    fetch1: Callable

    @property
    def _key_source(self):
        return self.dataset_table() & dict(dataset_fn=self.dataset_fn)

    def make(self, key: Key, get_mappings: Callable = integration.get_mappings) -> None:
        dataset_config = (self.dataset_table() & key).fetch1("dataset_config")
        mappings = get_mappings(dataset_config, key)
        self.insert(mappings)

    def get_output_selected_model(self, model: Module, key: Key) -> constrained_output_model:
        neuron_pos, session_id = (self & key).fetch1("neuron_position", "session_id")
        return self.constrained_output_model(model, neuron_pos, forward_kwargs=dict(data_key=session_id))


class MEIMethodMixin:
    definition = """
    # contains methods for generating MEIs and their configurations.
    method_fn                           : varchar(64)   # name of the method function
    method_hash                         : varchar(32)   # hash of the method config
    ---
    method_config                       : longblob      # method configuration object
    method_ts       = CURRENT_TIMESTAMP : timestamp     # UTZ timestamp at time of insertion
    method_comment                      : varchar(256)  # a short comment describing the method
    """

    insert1: Callable[[Mapping], None]
    __and__: Callable[[Mapping], MEIMethodMixin]
    fetch1: Callable

    seed_table = None
    import_func = staticmethod(integration.import_module)

    def add_method(self, method_fn: str, method_config: Mapping, comment: str = "") -> None:
        self.insert1(
            dict(
                method_fn=method_fn,
                method_hash=make_hash(method_config),
                method_config=method_config,
                method_comment=comment,
            )
        )

    def generate_mei(self, dataloaders: Dataloaders, model: Module, key: Key, seed: int) -> Dict[str, Any]:
        method_fn, method_config = (self & key).fetch1("method_fn", "method_config")
        method_fn = self.import_func(method_fn)
        mei, score, output = method_fn(dataloaders, model, method_config, seed)
        return dict(key, mei=mei, score=score, output=output)


class MEISeedMixin:
    definition = """
    # contains seeds used to make the MEI generation process reproducible
    mei_seed    : int   # MEI seed
    """


class MEITemplateMixin:
    definition = """
    # contains maximally exciting images (MEIs)
    -> self.method_table
    -> self.trained_model_table
    -> self.selector_table
    -> self.seed_table
    ---
    mei                 : attach@minio  # the MEI as a tensor
    score               : float         # some score depending on the used method function
    output              : attach@minio  # object returned by the method function
    """

    trained_model_table = None
    selector_table = None
    method_table = None
    seed_table = None
    model_loader_class = integration.ModelLoader
    save = staticmethod(torch.save)
    get_temp_dir = tempfile.TemporaryDirectory

    insert1: Callable[[Mapping], None]

    def __init__(self, *args, cache_size_limit: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_loader = self.model_loader_class(self.trained_model_table, cache_size_limit=cache_size_limit)

    def make(self, key: Key) -> None:
        dataloaders, model = self.model_loader.load(key=key)
        seed = (self.seed_table() & key).fetch1("mei_seed")
        output_selected_model = self.selector_table().get_output_selected_model(model, key)
        mei_entity = self.method_table().generate_mei(dataloaders, output_selected_model, key, seed)
        self._insert_mei(mei_entity)

    def _insert_mei(self, mei_entity: Dict[str, Any]) -> None:
        """Saves the MEI to a temporary directory and inserts the prepared entity into the table."""
        with self.get_temp_dir() as temp_dir:
            for name in ("mei", "output"):
                self._save_to_disk(mei_entity, temp_dir, name)
            self.insert1(mei_entity)

    def _save_to_disk(self, mei_entity: Dict[str, Any], temp_dir: str, name: str) -> None:
        data = mei_entity.pop(name)
        filename = name + "_" + self._create_random_filename() + ".pth.tar"
        filepath = os.path.join(temp_dir, filename)
        self.save(data, filepath)
        mei_entity[name] = filepath

    @staticmethod
    def _create_random_filename(length: Optional[int] = 32) -> str:
        return "".join(choice(ascii_letters) for _ in range(length))
