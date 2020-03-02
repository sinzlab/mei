from nnfabrik.utility.dj_helpers import make_hash
from . import integration


class CSRFV1SelectorHandler:
    def __init__(self, table):
        self.table = table

    def make(self, key):
        dataset_config = (self.table.dataset_table & key).fetch1("dataset_config")
        mappings = integration.get_mappings(dataset_config, key)
        self.table.insert(mappings)

    def get_output_selected_model(self, model, key):
        neuron_pos, session_id = (self.table & key).fetch1("neuron_position", "session_id")
        return integration.get_output_selected_model(neuron_pos, session_id, model)


class MEIMethodHandler:
    def __init__(self, mei_method_facade):
        self.mei_method_facade = mei_method_facade

    def add_method(self, method_fn, method_config):
        self.mei_method_facade.insert_method(
            dict(method_fn=method_fn, method_hash=make_hash(method_config), method_config=method_config)
        )

    def generate_mei(self, dataloader, model, key, import_func=integration.import_module):
        method_fn, method_config = self.mei_method_facade.fetch_method(key)
        method_fn = import_func(method_fn)
        mei, evaluations = method_fn(dataloader, model, method_config)
        return dict(key, evaluations=evaluations, mei=mei)
