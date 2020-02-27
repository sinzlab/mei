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
        self.trained_ensemble_table.insert1(key)

    def insert_members(self, members):
        self.member_table().insert(members)
