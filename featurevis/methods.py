import torch

from . import integration
from . import core


def gradient_ascent(dataloaders, model, config, import_func=None):
    if import_func is None:
        import_func = integration.import_module
    if not config["optim_kwargs"]:
        config["optim_kwargs"] = dict()
    for attribute in ("transform", "regularization", "gradient_f", "post_update"):
        if not config[attribute]:
            continue
        config[attribute] = import_func(config[attribute])
    mei_shape = integration.get_input_shape(dataloaders)
    initial_guess = torch.randn(1, *mei_shape[1:])
    mei, evaluations, _ = core.gradient_ascent(model, initial_guess, **config)
    return mei, evaluations
