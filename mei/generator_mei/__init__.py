"""This module contains methods used to generate MEIs."""

from typing import Tuple, Dict, Callable, Type

import torch
from torch import Tensor
from torch.nn import Module

from nnfabrik.utility.nn_helpers import get_dims_for_loader_dict
from .optimization import MEI

from ..import_helpers import resolve_object_path
from .tracking import Tracker


def prepare_object(config, _device="cpu", **kwargs):
    if not isinstance(config, dict):
        obj = config
    else:
        if isinstance(config["path"], str):
            object = resolve_object_path(config["path"])
        else:
            object = config["path"]
        obj = object(**config.get("kwargs", {}), **kwargs)
    if isinstance(obj, Module):
        obj.to(_device)
    return obj


def get_input_dimensions(dataloaders, get_dims):
    dataloaders_dimensions = list(get_dims(dataloaders["train"]).values())
    return list(dataloaders_dimensions[0].values())[0]


def generator_mei(
    input_shape,
    model: Module,
    seed: int,
    generator,
    optimizer,
    stopper,
    device="cuda",
    objectives=None,
    run=True,
    **config
) -> Tuple[Tensor, float, Dict]:
    """Generates a MEI using gradient ascent.

    The value corresponding to the "device" key must be either "cpu" or "cuda". The "transform",
    "regularization", "precondition" and "postprocessing" components are optional and can be omitted. All "kwargs" items
    in the config are optional and can be omitted as well. Furthermore the "objectives" item is optional and can be
    omitted. Example config:

        {
            "device": "cuda",
            "initial": {
                "path": "path.to.initial",
                "kwargs": {"initial_kwarg1": 0, "initial_kwarg2": 1},
            },
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

    Returns:
        The MEI, the final evaluation as a single float and the log of the tracker.
    """
    # input_shape = get_input_dimensions(dataloaders, get_dims)

    if objectives is None:
        objectives = []

    torch.manual_seed(seed)
    model.eval()
    model.to(device)

    input_generator = prepare_object(generator, input_shape=input_shape, _device=device)

    optimizer = prepare_object(
        optimizer, params=input_generator.parameters(), _device=device
    )

    stopper = prepare_object(stopper, _device=device)

    objectives = {
        obj_config["path"]
        if isinstance(obj_config, dict)
        else obj_config.__class__.__name__: prepare_object(obj_config, _device=device)
        for obj_config in objectives
    }

    tracker = Tracker(**objectives)

    # names of optinal components and extra inputs they should receive
    optional_components = {
        "transform": dict(input_shape=input_shape),
        "regularization": {},
        "precondition": {},
        "postprocessing": {},
    }

    optional = {
        opt: prepare_object(config[opt], _device=device, **extra)
        for opt, extra in optional_components.items()
        if opt in config
    }

    # instantiate MEI wrapper
    mei_obj = MEI(model, input_generator, optimizer, stopper, tracker, **optional)

    if run:
        final_evaluation, mei = mei_obj.optimize()
        return mei, final_evaluation, mei_obj.tracker.log
    else:
        return mei_obj