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
