from unittest.mock import MagicMock

import pytest
from torch import Tensor

from featurevis import domain


class TestInput:
    @pytest.fixture
    def input_(self, tensor):
        return domain.Input(tensor)

    @pytest.fixture
    def tensor(self):
        tensor = MagicMock(name="tensor", spec=Tensor)
        tensor.grad = "gradient"
        tensor.data = "data"
        tensor.detach.return_value.clone.return_value.cpu.return_value.squeeze.return_value = "extracted_tensor"
        tensor.__repr__ = MagicMock(name="repr", return_value="repr")
        return tensor

    def test_init(self, input_, tensor):
        assert input_.tensor is tensor

    def test_if_gradient_gets_enable_on_provided_tensor(self, input_, tensor):
        tensor.requires_grad_.assert_called_once_with()

    def test_gradient_property(self, input_, tensor):
        assert input_.gradient == "gradient"

    def test_gradient_setter(self, input_, tensor):
        input_.gradient = "new_gradient"
        assert tensor.grad == "new_gradient"

    def test_data_property(self, input_, tensor):
        assert input_.data == "data"

    def test_data_setter(self, input_, tensor):
        input_.data = "new_data"
        assert tensor.data == "new_data"

    def test_if_tensor_is_detached_when_extracted(self, input_, tensor):
        input_.extract()
        tensor.detach.assert_called_once_with()

    def test_if_tensor_is_cloned_when_extracted(self, input_, tensor):
        input_.extract()
        tensor.detach.return_value.clone.assert_called_once_with()

    def test_if_tensor_is_moved_to_cpu_when_extracted(self, input_, tensor):
        input_.extract()
        tensor.detach.return_value.clone.return_value.cpu.assert_called_once_with()

    def test_if_tensor_is_squeezed_when_extracted(self, input_, tensor):
        input_.extract()
        tensor.detach.return_value.clone.return_value.cpu.return_value.squeeze.assert_called_once_with()

    def test_if_extract_returns_correct_value(self, input_, tensor):
        assert input_.extract() == "extracted_tensor"

    def test_if_tensor_is_cloned_when_cloned(self, input_, tensor):
        input_.clone()
        tensor.clone.assert_called_once_with()

    def test_if_clone_returns_a_new_input_instance(self, input_):
        cloned = input_.clone()
        assert isinstance(cloned, domain.Input) and cloned is not input_

    def test_repr(self, input_, tensor):
        assert repr(input_) == f"Input({repr(tensor)})"
