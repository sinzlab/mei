import datajoint as dj

from nnfabrik.main import Dataset, schema
from . import handlers


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

    class Member(dj.Part):
        """Member table template."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.handler = handlers.TrainedEnsembleModelHandler.Member(self)

        @property
        def definition(self):
            return self.handler.definition

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handler = handlers.TrainedEnsembleModelHandler(self)

    @property
    def definition(self):
        return self.handler.definition

    def create_ensemble(self, *args, **kwargs):
        return self.handler.create_ensemble(*args, **kwargs)

    def load_model(self, *args, **kwargs):
        """Wrapper to preserve the interface of the trained model table."""
        return self.handler.load_model(*args, **kwargs)


class CSRFV1SelectorTemplate(dj.Computed):
    """CSRF V1 selector table template.

    To create a functional "CSRFV1Selector" table, create a new class that inherits from this template and decorate it
    with your preferred Datajoint schema. By default, the created table will point to the "Dataset" table in the
    Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting the class attribute called
    "dataset_table".
    """

    dataset_table = Dataset
    _key_source = dataset_table & dict(dataset_fn="csrf_v1")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handler = handlers.CSRFV1SelectorHandler(self)

    @property
    def definition(self):
        return self.handler.definition

    def make(self, *args, **kwargs):
        return self.handler.make(*args, **kwargs)

    def get_output_selected_model(self, *args, **kwargs):
        return self.handler.get_output_selected_model(*args, **kwargs)


@schema
class MEIMethod(dj.Lookup):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handler = handlers.MEIMethodHandler(self)

    @property
    def definition(self):
        return self.handler.definition

    def add_method(self, *args, **kwargs):
        return self.handler.add_method(*args, **kwargs)

    def generate_mei(self, *args, **kwargs):
        return self.handler.generate_mei(*args, **kwargs)


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

    def __init__(self, *args, cache_size_limit=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.handler = handlers.MEIHandler(self, cache_size_limit=cache_size_limit)

    @property
    def definition(self):
        return self.handler.definition

    def make(self, *args, **kwargs):
        return self.handler.make(*args, **kwargs)
