class TrainedEnsembleModelFacade:
    def __init__(self, trained_ensemble_table, member_table, dataset_table, trained_model_table):
        self.trained_ensemble_table = trained_ensemble_table
        self.member_table = member_table
        self.dataset_table = dataset_table
        self.trained_model_table = trained_model_table

    def properly_restricts(self, key):
        if len(self.dataset_table().proj() & key) == 1:
            return True
        return False

    def fetch_primary_dataset_key(self, key):
        return (self.dataset_table().proj() & key).fetch1()

    def fetch_trained_models(self, key=None):
        if key is None:
            key = dict()
        return (self.trained_model_table() & key).fetch(as_dict=True)

    def fetch_trained_models_primary_keys(self, key):
        return (self.trained_model_table().proj() & key).fetch(as_dict=True)

    def load_model(self, key):
        return self.trained_model_table().load_model(key=key)

    def insert_ensemble(self, key):
        self.trained_ensemble_table().insert1(key)

    def insert_members(self, members):
        self.member_table().insert(members)


class MEIMethodFacade:
    def __init__(self, method_table):
        self.method_table = method_table

    def insert_method(self, method):
        self.method_table().insert1(method)

    def fetch_method(self, key):
        return (self.method_table() & key).fetch1("method_fn", "method_config")


class MEIFacade:
    def __init__(self, mei_table, method_table, trained_model_table, selector_table):
        self.mei_table = mei_table
        self.method_table = method_table
        self.trained_model_table = trained_model_table
        self.selector_table = selector_table

    def get_output_selected_model(self, model, key):
        return self.selector_table().get_output_selected_model(model, key)

    def generate_mei(self, dataloaders, output_selected_model, key):
        return self.method_table().generate_mei(dataloaders, output_selected_model, key)

    def insert_mei(self, mei):
        self.mei_table().insert1(mei)
