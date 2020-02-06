import pickle

import datajoint as dj

from nnfabrik.main import Dataset, schema


class CSRFV1SelectorTemplate(dj.Computed):
    """CSRF V1 selector table template.

    To create a functional "CSRFV1Selector" table, create a new class that inherits from this template and decorate it
    with your preferred Datajoint schema. By default, the created table will point to the "Dataset" table in the
    Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting the class attribute called
    "dataset_table".
    """

    dataset_table = Dataset

    definition = """
        # contains information that can be used to map a neuron's id to its corresponding integer position in the output
        # of the model. 
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
