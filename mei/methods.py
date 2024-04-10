"""This module contains methods used to generate MEIs."""

from typing import Callable, Dict, Tuple, Type

import torch
from nnfabrik.utility.nn_helpers import get_dims_for_loader_dict
from torch import Tensor
from torch.nn import Module

from . import optimization
from .import_helpers import import_object
from .tracking import Tracker


def get_input_dimensions(dataloaders, get_dims, data_key=None):
    if data_key is None or data_key not in dataloaders["train"]:
        dataloaders_dimensions = list(get_dims(dataloaders["train"]).values())
        return list(dataloaders_dimensions[0].values())[0]
    else:
        dimensions_dict = get_dims(dataloaders["train"])[data_key]
        in_key = "inputs" if "inputs" in dimensions_dict else "images"
        return dimensions_dict[in_key]


def gradient_ascent(
    model: Module,
    config: Dict,
    seed: int,
    shape: tuple[int] = None,
    dataloaders: Dict = None,
    set_seed: Callable = torch.manual_seed,
    get_dims: Callable = get_dims_for_loader_dict,
    mei_class: Type = optimization.MEI,
    import_func: Callable = import_object,
    optimize_func: Callable = optimization.optimize,
    tracker_cls: Type[Tracker] = Tracker,
) -> Tuple[Tensor, float, Dict]:
    """Generates a MEI using gradient ascent.

    The value corresponding to the "device" key must be either "cpu" or "cuda". The "transform",
    "regularization", "precondition" and "postprocessing" components are optional and can be omitted. All "kwargs" items
    in the config are optional and can be omitted as well. Furthermore, the "objectives" item is optional and can be
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
        model: Callable object that will receive a tensor and must return a tensor containing a single float.
        config: Configuration dictionary. See above for an explanation and example.
        seed: Integer used to make the MEI generation process reproducible.
        shape: Shape of the input, if the shape is given, the dataloaders are not necessary.
        If dataloaders are not None, the shape will be inferred from there
        dataloaders: NNFabrik-style dataloader dictionary.
        set_seed: Callable for setting random seeds in torch.
        get_dims: Callable for inferring the dimensions from the dataloaders. If dataloaders are None can be None.
        mei_class: MEI class to use
        import_func: Callable for importing a class based on location.
        optimize_func: Callable to be used for optimization of the MEI.
        tracker_cls: Tracker to use.

    Returns:
        The MEI, the final evaluation as a single float and the log of the tracker.
    """
    if (dataloaders is None) and (shape is None):
        raise ValueError("Must provide either dataloader or shape")

    if dataloaders is not None:
        shape = get_input_dimensions(dataloaders, get_dims)

    for component_name, component_config in config.items():
        if component_name in (
            "device",
            "objectives",
            "n_meis",
            "mei_shape",
            "model_forward_kwargs",
            "transparency",
            "transparency_weight",
            "inhibitory",
        ):
            continue
        if "kwargs" not in component_config:
            component_config["kwargs"] = dict()

    if "objectives" not in config:
        config["objectives"] = []
    else:
        for obj in config["objectives"]:
            if "kwargs" not in obj:
                obj["kwargs"] = dict()

    set_seed(seed)
    model.eval()
    model.to(config["device"])

    n_meis = config.get("n_meis", 1)
    model_forward_kwargs = config.get("model_forward_kwargs", dict())
    model.forward_kwargs.update(model_forward_kwargs)

    create_initial_guess = import_func(config["initial"]["path"], config["initial"]["kwargs"])
    initial_guess = create_initial_guess(n_meis, *shape[1:]).to(config["device"])  # (1*1*h*w)

    transparency = config.get("transparency", None)
    if transparency:
        initial_alpha = (torch.ones(n_meis, 1, *shape[2:]) * 0.5).to(config["device"])
        # add transparency by concatenate alpha channel
        initial_guess = torch.cat((initial_guess, initial_alpha), dim=1)
    transparency_weight = config.get("transparency_weight", 1.0)
    inhibitory = config.get("inhibitory", None)

    optimizer = import_func(config["optimizer"]["path"], dict(params=[initial_guess], **config["optimizer"]["kwargs"]))
    stopper = import_func(config["stopper"]["path"], config["stopper"]["kwargs"])

    objectives = {o["path"]: import_func(o["path"], o["kwargs"]) for o in config["objectives"]}
    tracker = tracker_cls(**objectives)

    optional_names = ("transform", "regularization", "precondition", "postprocessing", "background")
    optional = {n: import_func(config[n]["path"], config[n]["kwargs"]) for n in optional_names if n in config}
    mei = mei_class(
        model,
        initial=initial_guess,
        optimizer=optimizer,
        transparency=transparency,
        inhibitory=inhibitory,
        transparency_weight=transparency_weight,
        **optional
    )

    final_evaluation, mei = optimize_func(mei, stopper, tracker)
    return mei, final_evaluation, tracker.log
