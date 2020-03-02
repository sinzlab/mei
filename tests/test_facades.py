from unittest.mock import MagicMock

from featurevis import facades


class TestMEIMethodFacade:
    def test_that_method_table_is_correctly_called_when_inserting_method(self):
        method_table = MagicMock()

        facade = facades.MEIMethodFacade(method_table)
        facade.insert_method("method")

        method_table.return_value.insert1.assert_called_once_with("method")

    def test_that_method_table_is_correctly_called_when_fetching_method(self):
        method_table = MagicMock()

        facade = facades.MEIMethodFacade(method_table)
        facade.fetch_method("key")

        method_table.return_value.__and__.assert_called_once_with("key")
        method_table.return_value.__and__.return_value.fetch1.assert_called_once_with("method_fn", "method_config")

    def test_that_method_is_returned(self):
        method_table = MagicMock()
        method_table.return_value.__and__.return_value.fetch1.return_value = "method"

        facade = facades.MEIMethodFacade(method_table)
        method = facade.fetch_method("key")

        assert method == "method"
