class MEIMethodFacade:
    def __init__(self, method_table):
        self.method_table = method_table

    def insert_method(self, method):
        self.method_table().insert1(method)

    def fetch_method(self, key):
        return (self.method_table() & key).fetch1("method_fn", "method_config")
