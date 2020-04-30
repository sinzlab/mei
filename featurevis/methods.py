from typing import Tuple, Dict, Callable, Type

import torch
from torch import Tensor
from torch.nn import Module

from nnfabrik.utility.nn_helpers import get_dims_for_loader_dict
from . import optimization
from .import_helpers import import_object
from .domain import Input
from .tracking import Tracker


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
    tracker_cls: Type[Tracker] = Tracker,
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
        tracker_cls: For testing purposes.

    Returns:
        The MEI, the final evaluation as a single float and the log of the tracker.
    """
    set_seed(seed)
    model.eval()
    model.to(config["device"])
    shape = get_input_dimensions(dataloaders, get_dims)
    initial_guess = create_initial_guess(1, *shape[1:], device=config["device"])

    optimizer = import_func(config["optimizer"], dict(params=[initial_guess], **config["optimizer_kwargs"]))
    stopper = import_func(config["stopper"], config["stopper_kwargs"])

    objectives = {o: import_func(o, ks) for o, ks in config["objectives"].items()}
    tracker = tracker_cls(**objectives)

    optional_names = ("transform", "regularization", "precondition", "postprocessing")
    optional = {n: import_func(config[n], config[n + "_kwargs"]) for n in optional_names if config[n]}

    mei = mei_class(model, input_cls(initial_guess), optimizer, **optional)

    final_evaluation, mei = optimize_func(mei, stopper, tracker)
    return mei, final_evaluation, tracker.log
