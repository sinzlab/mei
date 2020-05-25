from functools import partial
from unittest.mock import MagicMock

import pytest

from mei import import_helpers


class TestImportObject:
    @pytest.fixture
    def import_object(self, split_func, import_func):
        return partial(import_helpers.import_object, split_func=split_func, import_func=import_func)

    @pytest.fixture
    def split_func(self):
        return MagicMock(name="split_module_name", return_value=("package.module", "class"))

    @pytest.fixture
    def import_func(self, imported_object):
        return MagicMock(name="dynamic_import", return_value=imported_object)

    @pytest.fixture
    def imported_object(self):
        return MagicMock(name="imported_object", return_value="final_object")

    def test_if_split_func_is_called_correctly(self, import_object, split_func):
        import_object("package.module.callable")
        split_func.assert_called_once_with("package.module.callable")

    def test_if_import_func_is_called_correctly(self, import_object, import_func):
        import_object("package.module.callable")
        import_func.assert_called_once_with("package.module", "class")

    def test_if_imported_object_is_correctly_called_if_no_kwargs_are_provided(self, import_object, imported_object):
        import_object("package.module.callable")
        imported_object.assert_called_once_with()

    def test_if_imported_object_is_correctly_called_if_kwargs_are_provided(self, import_object, imported_object):
        import_object("package.module.callable", object_kwargs=dict(a=1, b=2))
        imported_object.assert_called_once_with(a=1, b=2)

    def test_if_object_returned_by_imported_object_is_returned(self, import_object, imported_object):
        obj = import_object("package.module.callable")
        assert obj == "final_object"

    def test_if_value_error_is_raised_if_provided_path_is_not_absolute(self, import_object, split_func):
        split_func.return_value = ("", "callable")
        with pytest.raises(ValueError):
            import_object("callable")

    def test_if_value_error_is_raised_if_module_not_found(self, import_object, import_func):
        import_func.side_effect = ModuleNotFoundError
        with pytest.raises(ValueError):
            import_object("package.module.callable")

    def test_if_value_error_is_raised_if_object_not_in_module(self, import_object, import_func):
        import_func.side_effect = AttributeError
        with pytest.raises(ValueError):
            import_object("package.module.callable")

    def test_if_value_error_is_raised_if_imported_object_is_not_callable(self, import_object, imported_object):
        imported_object.side_effect = TypeError
        with pytest.raises(ValueError):
            import_object("package.module.callable")
