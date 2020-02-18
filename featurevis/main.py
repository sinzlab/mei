import tempfile
import os

import datajoint as dj
import torch

from nnfabrik.main import Dataset, schema
from nnfabrik.utility.nn_helpers import get_dims_for_loader_dict
from nnfabrik.utility.nnf_helper import split_module_name, dynamic_import
from nnfabrik.utility.dj_helpers import make_hash
from .core import gradient_ascent
from . import table_funcs


class TrainedEnsembleModelTemplate(dj.Manual):
    """TrainedEnsembleModel table template.

    To create a functional "TrainedEnsembleModel" table, create a new class that inherits from this template and
    decorate it with your preferred Datajoint schema. Next assign the trained model table of your choosing to the class
    variable called "trained_model_table". By default the created table will point to the "Dataset" table in the
    Datajoint schema called "nnfabrik.main". This behaviour can be changed by overwriting the class attribute called
    "dataset_table".
    """

    dataset_table = Dataset
    trained_model_table = None

    definition = """
    # contains ensemble ids
    -> self.dataset_table
    ensemble_id : tinyint unsigned  # the ensemble id
    """

    class Member(dj.Part):
        """Member table template."""

        definition = """
        # contains assignments of trained models to a specific ensemble id
        -> master
        -> master.trained_model_table
        """

    def load_model(self, key=None):
        """Wrapper to preserve the interface of the trained model table."""
        return table_funcs.load_ensemble_model(self.Member, self.trained_model_table, key=key)


class CSRFV1SelectorTemplate(dj.Computed):
    """CSRF V1 selector table template.

    To create a functional "CSRFV1Selector" table, create a new class that inherits from this template and decorate it
    with your preferred Datajoint schema. By default, the created table will point to the "Dataset" table in the
    Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting the class attribute called
    "dataset_table".
    """

    dataset_table = Dataset

    definition = """
    # contains information that can be used to map a neuron's id to its corresponding integer position in the output of
    # the model. 
    -> self.dataset_table
    neuron_id       : smallint unsigned # unique neuron identifier
    ---
    neuron_position : smallint unsigned # integer position of the neuron in the model's output 
    session_id      : varchar(13)       # unique session identifier
    """

    _key_source = Dataset & dict(dataset_fn="csrf_v1")

    def make(self, key):
        dataset_config = (Dataset & key).fetch1("dataset_config")
        mappings = table_funcs.get_mappings(dataset_config, key)
        self.insert(mappings)

    def get_output_selected_model(self, model, key):
        neuron_pos, session_id = (self & key).fetch1("neuron_position", "session_id")
        return table_funcs.get_output_selected_model(neuron_pos, session_id, model)


@schema
class MEIMethod(dj.Lookup):
    definition = """
    # contains parameters used in MEI generation
    method_id                   : tinyint unsigned      # integer that uniquely identifies a set of parameter values
    ---
    transform           = NULL  : varchar(64)           # differentiable function that transforms the MEI before sending
                                                        # it to through the model
    regularization      = NULL  : varchar(64)           # differentiable function used for regularization
    gradient_f          = NULL  : varchar(64)           # non-differentiable function that receives the gradient of the
                                                        # MEI and outputs a preconditioned gradient
    post_update         = NULL  : varchar(64)           # non-differentiable function applied to the MEI after each 
                                                        # gradient update
    step_size           = 0.1   : float                 # size of the step size to give every iteration
    optim_name          = "SGD" : enum("SGD", "Adam")   # optimizer to be used
    optim_kwargs        = NULL  : longblob              # dictionary containing keyword arguments for the optimizer
    num_iterations      = 1000  : smallint unsigned     # number of gradient ascent steps
    """

    def generate_mei(self, dataloaders, model, key):
        """Generates a MEI.

        Args:
            dataloaders: A dictionary of dataloaders.
            model: A PyTorch module.
            key: A dictionary used to restrict this table to a single entry.

        Returns:
            A dictionary containing the MEI ready for insertion into the MEI table.
        """
        method_id, method = (self & key).get_mei_method()
        input_shape = self._get_input_shape(dataloaders)
        initial_guess = torch.randn(1, *input_shape[1:])
        mei, evaluations, _ = gradient_ascent(model, initial_guess, **method)
        return dict(key, evaluations=evaluations, mei=mei)

    def get_mei_method(self):
        """Fetches a set of MEI generation parameters and makes them ready to be used in the MEI table.

        This function assumes that the table is already restricted to one entry when it is called.
        """
        method = self.fetch1()
        if not method["optim_kwargs"]:
            method["optim_kwargs"] = dict()
        for attribute in ("transform", "regularization", "gradient_f", "post_update"):
            if not method[attribute]:
                continue
            abs_module_path, function_name = split_module_name(method[attribute])
            method[attribute] = dynamic_import(abs_module_path, function_name)
        return method.pop("method_id"), method

    @staticmethod
    def _get_input_shape(dataloaders):
        """Gets the shape of the input that the model expects from the dataloaders."""
        return list(get_dims_for_loader_dict(dataloaders["train"]).values())[0]["inputs"]


class MEITemplate(dj.Computed):
    """MEI table template.

    To create a functional "MEI" table, create a new class that inherits from this template and decorate it with your
    preferred Datajoint schema. Next assign your trained model (or trained ensemble model) and your selector table to
    the class variables called "trained_model_table" and "selector_table". By default, the created table will point to
    the "MEIMethod" table in the Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting
    the class attribute called "method_table".
    """

    method_table = MEIMethod
    trained_model_table = None
    selector_table = None

    definition = """
    # contains maximally exciting images (MEIs)
    -> self.method_table
    -> self.trained_model_table
    -> self.selector_table
    ---
    mei                 : attach@minio  # the MEI as a tensor
    evaluations         : longblob      # list of function evaluations at each iteration in the mei generation process 
    """

    def make(self, key):
        dataloaders, model = self.trained_model_table().load_ensemble_model(key=key)
        output_selected_model = self.selector_table().get_output_selected_model(model, key)
        mei_entity = self.method_table().generate_mei(dataloaders, output_selected_model, key)
        self._insert_mei(mei_entity)

    def _insert_mei(self, mei_entity):
        """Saves the MEI to a temporary directory and inserts the prepared entity into the table."""
        mei = mei_entity.pop("mei").squeeze()
        filename = make_hash(mei_entity) + ".pth.tar"
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, filename)
            torch.save(mei, filepath)
            mei_entity["mei"] = filepath
            self.insert1(mei_entity)
