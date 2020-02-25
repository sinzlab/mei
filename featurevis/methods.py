import torch

from nnfabrik.utility.nnf_helper import split_module_name, dynamic_import
from nnfabrik.utility.nn_helpers import get_dims_for_loader_dict
from . import core


def import_path(path):
    return dynamic_import(*split_module_name(path))


class GradientAscent:
    """Wrapper class around gradient ascent function.

    Attributes:
        import_object: Function used for importing functions given a path as a string.
        get_dims: Function used to get the input and output dimensions of all dataloaders in a dictionary.
        get_initial_guess: Function used to get the initial random guess for the gradient ascent function.
        ascend: Function used to do the actual gradient ascent.
    """

    def __init__(
        self,
        import_object=import_path,
        get_dims=get_dims_for_loader_dict,
        get_initial_guess=torch.randn,
        ascend=core.gradient_ascent,
    ):
        self.import_object = import_object
        self.get_dims = get_dims
        self.get_initial_guess = get_initial_guess
        self.ascend = ascend

    def __call__(self, dataloaders, model, config):
        """Generates MEIs.

        Args:
            dataloaders: NNFabrik style dataloader dictionary.
            model: Callable object that returns a single number.
            config: Dictionary of arguments for the gradient ascent function.

        Returns:
            The MEI as a tensor and a list of model evaluations at each step of the gradient ascent process.
        """
        config = self._prepare_config(config)
        mei_shape = self._get_input_dimensions(dataloaders)
        initial_guess = self.get_initial_guess(1, *mei_shape[1:])
        mei, evaluations, _ = self.ascend(model, initial_guess, **config)
        return mei, evaluations

    def _prepare_config(self, config):
        config = self._prepare_optim_kwargs(config)
        return self._import_functions(config)

    @staticmethod
    def _prepare_optim_kwargs(config):
        if not config["optim_kwargs"]:
            config["optim_kwargs"] = dict()
        return config

    def _import_functions(self, config):
        for attribute in ("transform", "regularization", "gradient_f", "post_update"):
            if not config[attribute]:
                continue
            config[attribute] = self.import_object(config[attribute])
        return config

    def _get_input_dimensions(self, dataloaders):
        return list(self.get_dims(dataloaders["train"]).values())[0]["inputs"]
