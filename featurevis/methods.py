"""This module contains methods used to generate MEIs."""

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


def gradient_ascent(
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

    The value corresponding to the "device" key must be either "cpu" or "cuda". The values corresponding to the
    "transform", "regularization", "precondition" and "postprocessing" keys should be "None" if the corresponding
    component is not used. All values corresponding to keys called "kwargs" should be "None" if there are no keyword
    arguments to pass to the corresponding component. The value corresponding to the "objectives" key should be "None"
    if no objectives are to be tracked. Example config:

        {
            "device": "cuda",
            "optimizer": {
                "path": "path.to.optimizer",
                "kwargs": {"optimizer_kwarg1": 0, "optimizer_kwarg2": 1},
            },
            "stopper": {
                "path": "path.to.stopper",
                "kwargs": {"stopper_kwarg1": 0, "stopper_kwarg2": 0},
            },
            "transform": {
                "path": "path.to.transform",
                "kwargs": {"transform_kwarg1": 0, "transform_kwarg2": 1},
            },
            "regularization": {
                "path": "path.to.regularization",
                "kwargs": {"regularization_kwarg1": 0, "regularization_kwarg2": 1},
            },
            "precondition": {
                "path": "path.to.precondition",
                "kwargs": {"precondition_kwarg1": 0, "precondition_kwarg2": 1},
            },
            "postprocessing": {
                "path": "path.to.postprocessing",
                "kwargs": {"postprocessing_kwarg1": 0, "postprocessing_kwarg2": 1},
            },
            "objectives": [
                {"path": "path.to.objective1", "kwargs": {"objective1_kwarg1": 0, "objective1_kwarg2": 1}},
                {"path": "path.to.objective2", "kwargs": {"objective2_kwarg1": 0, "objective2_kwarg2": 1}},
            ],
        }

    Args:
        dataloaders: NNFabrik-style dataloader dictionary.
        model: Callable object that will receive a tensor and must return a tensor containing a single float.
        config: Configuration dictionary. See above for an explanation and example.
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
    for component_name, component_config in config.items():
        if component_name in ("device", "objectives"):
            continue
        if component_config is not None and component_config["kwargs"] is None:
            component_config["kwargs"] = dict()
    if config["objectives"] is None:
        config["objectives"] = []
    else:
        for obj in config["objectives"]:
            if obj["kwargs"] is None:
                obj["kwargs"] = dict()

    set_seed(seed)
    model.eval()
    model.to(config["device"])
    shape = get_input_dimensions(dataloaders, get_dims)
    initial_guess = create_initial_guess(1, *shape[1:], device=config["device"])

    optimizer = import_func(config["optimizer"]["path"], dict(params=[initial_guess], **config["optimizer"]["kwargs"]))
    stopper = import_func(config["stopper"]["path"], config["stopper"]["kwargs"])

    objectives = {o["path"]: import_func(o["path"], o["kwargs"]) for o in config["objectives"]}
    tracker = tracker_cls(**objectives)

    optional_names = ("transform", "regularization", "precondition", "postprocessing")
    optional = {n: import_func(config[n]["path"], config[n]["kwargs"]) for n in optional_names if config[n]}

    mei = mei_class(model, input_cls(initial_guess), optimizer, **optional)

    final_evaluation, mei = optimize_func(mei, stopper, tracker)
    return mei, final_evaluation, tracker.log
