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


class TestMeiFacade:
    def test_that_call_to_get_output_selected_model_is_correct(self):
        selector_table_instance = MagicMock()
        selector_table = MagicMock(return_value=selector_table_instance)

        facade = facades.MEIFacade(None, None, None, selector_table)
        facade.get_output_selected_model("model", "key")

        selector_table_instance.get_output_selected_model.assert_called_once_with("model", "key")

    def test_that_output_selected_model_is_returned(self):
        selector_table_instance = MagicMock()
        selector_table_instance.get_output_selected_model.return_value = "model"
        selector_table = MagicMock(return_value=selector_table_instance)

        facade = facades.MEIFacade(None, None, None, selector_table)
        model = facade.get_output_selected_model("model", "key")

        assert model == "model"

    def test_that_call_to_generate_mei_is_correct(self):
        method_table_instance = MagicMock()
        method_table = MagicMock(return_value=method_table_instance)

        facade = facades.MEIFacade(None, method_table, None, None)
        facade.generate_mei("dataloaders", "output_selected_model", "key")

        method_table_instance.generate_mei.assert_called_once_with("dataloaders", "output_selected_model", "key")

    def test_that_generated_mei_is_returned(self):
        method_table_instance = MagicMock()
        method_table_instance.generate_mei.return_value = "mei"
        method_table = MagicMock(return_value=method_table_instance)

        facade = facades.MEIFacade(None, method_table, None, None)
        mei = facade.generate_mei("dataloaders", "output_selected_model", "key")

        assert mei == "mei"

    def test_that_call_to_mei_table_is_correct(self):
        mei_table_instance = MagicMock()
        mei_table = MagicMock(return_value=mei_table_instance)

        facade = facades.MEIFacade(mei_table, None, None, None)
        facade.insert_mei("mei")

        mei_table_instance.insert1.assert_called_once_with("mei")
