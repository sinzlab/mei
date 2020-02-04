import pickle

import datajoint as dj

from nnfabrik.main import Dataset


class CSRFV1SelectorTemplate(dj.Computed):
    """CSRF V1 selector table template.

    To create a functional "CSRFV1Selector" table, create a new class that inherits from this template and decorate it
    with your preferred Datajoint schema. By default, the created table will point to the "Dataset" table in the
    Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting the class attribute called
    "dataset_table".
    """

    dataset_table = Dataset

    @property
    def definition(self):
        return """
        # contains information that can be used to map a neuron's id to its corresponding integer position in the output
        # of the model. 
        -> self.dataset_table
        neuron_id       : smallint unsigned # unique neuron identifier
        ---
        neuron_position : smallint unsigned # integer position of the neuron in the model's output 
        """

    _key_source = Dataset & dict(dataset_fn="csrf_v1")

    def make(self, key):
        dataset_config = (Dataset & key).fetch1("dataset_config")
        entities = []
        for datafile_path in dataset_config["datafiles"]:
            with open(datafile_path, "rb") as datafile:
                data = pickle.load(datafile)
            for neuron_pos, neuron_id in enumerate(data["unit_indices"]):
                entities.append(dict(key, neuron_id=neuron_id, neuron_position=neuron_pos))
        self.insert(entities)

    def get_select_neuron(self, neuron_id):
        """Creates a function that can be used to select the output corresponding to a uniquely identified neuron from
        the whole output of the model.

        Args:
            neuron_id: An integer that can be used to uniquely identify a neuron.

        Returns:
            A function that takes the model's output, selects the part corresponding to the uniquely identified neuron
            and returns said part.
        """
        neuron_position = (self & dict(neuron_id=neuron_id)).fetch1("neuron_position")

        def select_neuron(model_output):
            return model_output[:, neuron_position]

        return select_neuron
