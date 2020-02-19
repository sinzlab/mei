import pickle

import torch

from nnfabrik.utility.nn_helpers import get_dims_for_loader_dict
from nnfabrik.utility.nnf_helper import split_module_name, dynamic_import


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


def prepare_mei_method(method, import_func=None):
    if import_func is None:
        import_func = import_module
    if not method["optim_kwargs"]:
        method["optim_kwargs"] = dict()
    for attribute in ("transform", "regularization", "gradient_f", "post_update"):
        if not method[attribute]:
            continue
        method[attribute] = import_func(method[attribute])
    method.pop("method_id")
    return method


def import_module(path):
    return dynamic_import(*split_module_name(path))
