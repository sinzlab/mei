import pickle
import tempfile
import os

import datajoint as dj
import torch

from nnfabrik.main import Dataset, schema
from nnfabrik.utility.nn_helpers import get_dims_for_loader_dict
from nnfabrik.utility.nnf_helper import split_module_name, dynamic_import
from nnfabrik.utility.dj_helpers import make_hash
from .core import gradient_ascent


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
        entities = []
        for datafile_path in dataset_config["datafiles"]:
            with open(datafile_path, "rb") as datafile:
                data = pickle.load(datafile)
            for neuron_pos, neuron_id in enumerate(data["unit_indices"]):
                entities.append(
                    dict(key, neuron_id=neuron_id, neuron_position=neuron_pos, session_id=data["session_id"])
                )
        self.insert(entities)

    def get_output_selected_model(self, model, neuron_id):
        """Creates a version of the model that has its output selected down to a single uniquely identified neuron.

        Args:
            model: A PyTorch module that can be called with a keyword argument called "data_key". The output of the
                module is expected to be a two dimensional Torch tensor where the first dimension corresponds to the
                batch size and the second to the number of neurons.
            neuron_id: An integer that can be used to uniquely identify a neuron.

        Returns:
            A function that takes the model input(s) as parameter(s) and returns the model output corresponding to the
            selected neuron.
        """
        neuron_pos, session_id = (self & dict(neuron_id=neuron_id)).fetch1("neuron_position", "session_id")

        def output_selected_model(x, *args, **kwargs):
            output = model(x, *args, data_key=session_id, **kwargs)
            return output[:, neuron_pos]

        return output_selected_model


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
        dataloaders, model = self.trained_model_table().load_model(key=key)
        neuron_id = (self.selector_table & key).fetch1("neuron_id")
        method_id, method = (self.method_table & key).get_mei_method()
        input_shape = self._get_input_shape(dataloaders)
        initial_guess = torch.randn(1, *input_shape[1:])
        output_selected_model = self.selector_table().get_output_selected_model(model, neuron_id)
        mei, evaluations, _ = gradient_ascent(output_selected_model, initial_guess, **method)
        mei_entity = dict(key, neuron_id=neuron_id, method_id=method_id, evaluations=evaluations, mei=mei)
        self._insert_mei(mei_entity)

    @staticmethod
    def _get_input_shape(dataloaders):
        """Gets the shape of the input that the model expects from the dataloaders."""
        return list(get_dims_for_loader_dict(dataloaders["train"]).values())[0]["inputs"]

    def _insert_mei(self, mei_entity):
        """Saves the MEI to a temporary directory and inserts the prepared entity into the table."""
        mei = mei_entity.pop("mei").squeeze()
        filename = make_hash(mei_entity) + ".pth.tar"
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, filename)
            torch.save(mei, filepath)
            mei_entity["mei"] = filepath
            self.insert1(mei_entity)


class TrainedEnsembleModelTemplate(dj.Manual):
    """TrainedEnsembleModel table template.

    To create a functional "TrainedEnsembleModel" table, create a new class that inherits from this template and
    decorate it with your preferred Datajoint schema. By default the created table will point to the "Dataset" table in
    the Datajoint schema called "nnfabrik.main". This behaviour can be changed by overwriting the class attribute called
    "dataset_table".
    """

    dataset_table = Dataset

    definition = """
    # contains ensemble ids
    -> self.dataset_table
    ensemble_id : tinyint unsigned  # the ensemble id
    """

    class Member(dj.Part):
        """Member part table template.

        To create a functional "Member" table, create a new class inside your "TrainedEnsembleModel" table that inherits
        from this template and is called "Member". Then you assign your "TrainedModel" table to the class variable
        called "trained_model_table".
        """

        trained_model_table = None

        definition = """
        # contains assignments of trained models to a specific ensemble id
        -> master
        -> self.trained_model_table
        """

        def load_model(self, key=None):
            """Wrapper around the "load_model" method in the trained model table."""
            dataloaders, model = self.trained_model_table().load_model(key=key)
            return dataloaders, model

    def load_model(self, key=None):
        """Loads an ensemble model.

        Args:
            key: A dictionary used to restrict the member part table.

        Returns:
            A function that has the model's input as parameters and returns the mean output across the individual models
            in the ensemble.
        """

        def ensemble_model(x, *args, **kwargs):
            outputs = [m(x, *args, **kwargs) for m in models]
            mean_output = torch.stack(outputs, dim=0).mean(dim=0)
            return mean_output

        if key:
            query = self.Member() & key
        else:
            query = self.Member()
        model_keys = query.fetch(as_dict=True)
        dataloaders, models = tuple(list(x) for x in zip(*[self.Member().load_model(key=k) for k in model_keys]))
        return dataloaders[0], ensemble_model
