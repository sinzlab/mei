import datajoint as dj

from nnfabrik.main import Dataset, schema
from . import handlers
from . import facades
from . import tables


class TrainedEnsembleModelTemplate(tables.TrainedEnsembleModelTemplate, dj.Manual):
    """TrainedEnsembleModel table template.

    To create a functional "TrainedEnsembleModel" table, create a new class that inherits from this template and
    decorate it with your preferred Datajoint schema. Next assign the trained model table of your choosing to the class
    variable called "trained_model_table". By default the created table will point to the "Dataset" table in the
    Datajoint schema called "nnfabrik.main". This behaviour can be changed by overwriting the class attribute called
    "dataset_table".
    """

    dataset_table = Dataset

    class Member(tables.TrainedEnsembleModelTemplate.Member, dj.Part):
        """Member table template."""


class CSRFV1SelectorTemplate(dj.Computed):
    """CSRF V1 selector table template.

    To create a functional "CSRFV1Selector" table, create a new class that inherits from this template and decorate it
    with your preferred Datajoint schema. By default, the created table will point to the "Dataset" table in the
    Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting the class attribute called
    "dataset_table".
    """

    definition = """
    # contains information that can be used to map a neuron's id to its corresponding integer position in the output of
    # the model. 
    -> self.dataset_table
    neuron_id       : smallint unsigned # unique neuron identifier
    ---
    neuron_position : smallint unsigned # integer position of the neuron in the model's output 
    session_id      : varchar(13)       # unique session identifier
    """

    dataset_table = Dataset
    _key_source = dataset_table & dict(dataset_fn="csrf_v1")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handler = handlers.CSRFV1SelectorHandler(self)

    def make(self, *args, **kwargs):
        return self.handler.make(*args, **kwargs)

    def get_output_selected_model(self, *args, **kwargs):
        return self.handler.get_output_selected_model(*args, **kwargs)


@schema
class MEIMethod(dj.Lookup):
    definition = """
    # contains methods for generating MEIs and their configurations.
    method_fn                           : varchar(64)   # name of the method function
    method_hash                         : varchar(32)   # hash of the method config
    ---
    method_config                       : longblob      # method configuration object
    method_ts       = CURRENT_TIMESTAMP : timestamp     # UTZ timestamp at time of insertion
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handler = handlers.MEIMethodHandler(facades.MEIMethodFacade(self.__class__))

    def add_method(self, *args, **kwargs):
        return self.handler.add_method(*args, **kwargs)

    def generate_mei(self, *args, **kwargs):
        return self.handler.generate_mei(*args, **kwargs)


class MEITemplate(tables.MEITemplate, dj.Computed):
    """MEI table template.

    To create a functional "MEI" table, create a new class that inherits from this template and decorate it with your
    preferred Datajoint schema. Next assign your trained model (or trained ensemble model) and your selector table to
    the class variables called "trained_model_table" and "selector_table". By default, the created table will point to
    the "MEIMethod" table in the Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting
    the class attribute called "method_table".
    """

    method_table = MEIMethod
