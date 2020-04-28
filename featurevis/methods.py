from typing import Tuple, Dict, Callable, Type

import torch
from torch import Tensor
from torch.nn import Module

from nnfabrik.utility.nnf_helper import split_module_name, dynamic_import
from nnfabrik.utility.nn_helpers import get_dims_for_loader_dict
from . import core
from . import optimization
from .import_helpers import import_object
from .domain import Input


def import_path(path):
    return dynamic_import(*split_module_name(path))


def ascend(model, initial_guess, config, ascending_func=core.gradient_ascent):
    """Wrapper around original gradient ascent used to package up the returned result."""
    mei, function_evaluations, regularization_terms = ascending_func(model, initial_guess, **config)
    output = dict(function_evaluations=function_evaluations)
    if isinstance(mei, list):
        mei = [m.cpu().squeeze() for m in mei]
        output["progression"] = mei
        mei = mei[-1]
    else:
        mei = mei.cpu().squeeze()
    if regularization_terms:
        output["regularization_terms"] = regularization_terms
    return mei, function_evaluations[-1], output


def gradient_ascent(
    dataloaders,
    model,
    config,
    seed,
    set_seed=torch.manual_seed,
    import_object=import_path,
    get_dims=get_dims_for_loader_dict,
    get_initial_guess=torch.randn,
    ascend_func=ascend,
):
    """Generates MEIs.

    Args:
        dataloaders: NNFabrik style dataloader dictionary.
        model: Callable object that returns a single number.
        config: Dictionary of arguments for the gradient ascent function.
        seed: Integer used to make the MEI generation process reproducible.
        set_seed: Function used to set the seed. For testing purposes.
        import_object: Function used to import functions given a path as a string. For testing purposes.
        get_dims: Function used to get the input and output dimensions for all dataloaders. For testing purposes.
        get_initial_guess: Function used to get the initial random guess for the gradient ascent. For testing purposes.
        ascend_func: Function used to do the actual ascending. For testing purposes.

    Returns:
        The MEI as a tensor and a list of model evaluations at each step of the gradient ascent process.
    """

    set_seed(seed)
    model.eval()
    config = prepare_config(config, import_object)
    mei_shape = get_input_dimensions(dataloaders, get_dims)
    device = config.pop("device")
    initial_guess = get_initial_guess(1, *mei_shape[1:], device=device)
    model.to(device)
    return ascend_func(model, initial_guess, config)


def prepare_config(config, import_object):
    config = prepare_optim_kwargs(config)
    return import_functions(config, import_object)


def prepare_optim_kwargs(config):
    if not config["optim_kwargs"]:
        config["optim_kwargs"] = dict()
    return config


def import_functions(config, import_object):
    for attribute in ("transform", "regularization", "gradient_f", "post_update"):
        if not config[attribute]:
            continue
        config[attribute] = import_object(config[attribute])
    return config


def get_input_dimensions(dataloaders, get_dims):
    return list(get_dims(dataloaders["train"]).values())[0]["inputs"]


def ascend_gradient(
    dataloaders: Dict,
    model: Module,
    config: Dict,
    seed: int,
    set_seed: Callable = torch.manual_seed,
    get_dims: Callable = get_dims_for_loader_dict,
    create_initial_guess: Callable = torch.randn,
    input_cls: Callable = Input,
    mei_class: Type = optimization.MEI,
    import_func: Callable = import_object,
    optimize_func: Callable = optimization.optimize,
) -> Tuple[Tensor, float, Dict]:
    """Generates a MEI using gradient ascent.

    Args:
        dataloaders: NNFabrik-style dataloader dictionary.
        model: Callable object that will receive a tensor and must return a tensor containing a single float.
        config: A dictionary containing the following keys: "optimizer", "optimizer_kwargs", "stopper",
            "stopper_kwargs", "transform", "transform_kwargs", "regularization", "regularization_kwargs",
            "precondition", "precondition_kwargs", "postprocessing", "postprocessing_kwargs", "device". The values
            corresponding to the "optimizer", "stopper", "transform" ,"regularization", "precondition" and
            "postprocessing" keys must be absolute paths pointing to the optimizer, stopper, transform ,regularization,
            precondition and postprocessing callables, respectively. The values corresponding to the "optimizer_kwargs",
            "stopper_kwargs", "transform_kwargs" ,"regularization_kwargs", "precondition_kwargs" and
            "postprocessing_kwargs" keys must be dictionaries containing keyword arguments with which the respective
            callables will be called.
            The value corresponding to the "device" key must be either "cuda" or "cpu".
            No transform will be used if the value belonging to the "transform" key is "None". The value belonging to
            the "transform_kwargs" key should also be "None" if that is the case. The same logic can be applied to the
            "regularization", "precondition" and "postprocessing" keys.
        seed: Integer used to make the MEI generation process reproducible.
        set_seed: For testing purposes.
        get_dims: For testing purposes.
        create_initial_guess: For testing purposes.
        input_cls: For testing purposes.
        mei_class: For testing purposes.
        import_func: For testing purposes.
        optimize_func: For testing purposes.

    Returns:
        The MEI, the final evaluation as a single float and an empty dictionary.
    """
    set_seed(seed)
    model.eval()
    model.to(config["device"])
    shape = get_input_dimensions(dataloaders, get_dims)
    initial_guess = create_initial_guess(1, *shape[1:], device=config["device"])

    optimizer = import_func(config["optimizer"], dict(params=[initial_guess], **config["optimizer_kwargs"]))
    stopper = import_func(config["stopper"], config["stopper_kwargs"])

    optional_names = ("transform", "regularization", "precondition", "postprocessing")
    optional = {n: import_func(config[n], config[n + "_kwargs"]) for n in optional_names if config[n]}

    mei = mei_class(model, input_cls(initial_guess), optimizer, **optional)

    final_evaluation, mei = optimize_func(mei, stopper)
    return mei, final_evaluation, dict()
