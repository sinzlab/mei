from unittest.mock import MagicMock
from typing import Type

import pytest
import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from mei import modules


@pytest.fixture
def module_mock():
    class ModuleMock(Module):
        __init__ = MagicMock(name="ModuleMock().__init__", return_value=None)

    return ModuleMock


class TestEnsembleModel:
    @pytest.fixture
    def ensemble_model(self, ensemble_model_cls, members):
        return ensemble_model_cls(*members)

    @pytest.fixture
    def ensemble_model_cls(self, module_container_cls):
        ensemble_model_cls = modules.EnsembleModel
        ensemble_model_cls._module_container_cls = module_container_cls
        return ensemble_model_cls

    @pytest.fixture
    def members(self):
        members = []
        for i in range(3):
            values = list(x + i * 3 for x in range(1, 4))
            member = MagicMock(name="member" + str(i + 1), return_value=torch.tensor([values], dtype=torch.float))
            member.__repr__ = MagicMock(return_value="member" + str(i + 1))
            members.append(member)
        return members

    @pytest.fixture
    def module_container_cls(self, members, module_container):
        return MagicMock(name="module_container_cls", spec=Type[ModuleList], return_value=module_container)

    @pytest.fixture
    def module_container(self, members):
        return [m for m in members]

    @pytest.fixture
    def ensemble_input(self):
        return MagicMock(name="ensemble_input", spec=Tensor)

    def test_if_ensemble_model_is_pytorch_module(self, ensemble_model_cls):
        assert issubclass(ensemble_model_cls, Module)

    def test_if_ensemble_model_initializes_super_class(self, ensemble_model_cls, module_mock):
        class EnsembleModelTestable(ensemble_model_cls, module_mock):
            pass

        EnsembleModelTestable()
        module_mock.__init__.assert_called_once_with()

    def test_if_module_container_class_is_correctly_initialized(self, ensemble_model, module_container_cls, members):
        module_container_cls.assert_called_once_with(tuple(members))

    def test_if_module_container_is_assigned_as_attribute(self, ensemble_model, module_container):
        assert ensemble_model.members == module_container

    def test_if_input_is_passed_to_ensemble_members(self, ensemble_model, members, ensemble_input):
        ensemble_model(ensemble_input, "arg", kwarg="kwarg")
        for member in members:
            member.assert_called_once_with(ensemble_input, "arg", kwarg="kwarg")

    def test_if_outputs_of_ensemble_members_is_correctly_averaged(self, ensemble_model, ensemble_input):
        output = ensemble_model(ensemble_input)
        assert torch.allclose(output, torch.tensor([4, 5, 6], dtype=torch.float))

    def test_repr(self, ensemble_model):
        assert str(ensemble_model) == "EnsembleModel(member1, member2, member3)"


class TestConstrainedOutputModel:
    @pytest.fixture
    def model(self):
        model = MagicMock(name="model", return_value=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]))
        model.__repr__ = MagicMock(name="__repr__", return_value="model")
        return model

    @pytest.fixture
    def model_input(self):
        return MagicMock(name="model_input", spec=Tensor)

    def test_if_constrained_output_model_is_pytorch_module(self):
        assert issubclass(modules.ConstrainedOutputModel, Module)

    def test_if_super_class_is_initialized(self, model, module_mock):
        class ConstrainedOutputModelTestable(modules.ConstrainedOutputModel, module_mock):
            pass

        ConstrainedOutputModelTestable(model, 0)
        module_mock.__init__.assert_called_once_with()

    def test_if_input_is_passed_to_model(self, model, model_input):
        constrained_model = modules.ConstrainedOutputModel(model, 0)
        constrained_model(model_input, "arg", kwarg="kwarg")
        model.assert_called_once_with(model_input, "arg", kwarg="kwarg")

    @pytest.mark.parametrize("constraint,expected", [(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0), (4, 5.0)])
    def test_if_output_constraint_is_correct(self, model, model_input, constraint, expected):
        constrained_model = modules.ConstrainedOutputModel(model, constraint)
        output = constrained_model(model_input)
        assert torch.allclose(output, torch.tensor([expected]))

    def test_if_forward_kwargs_are_passed_to_model(self, model, model_input):
        constrained_model = modules.ConstrainedOutputModel(model, 0, forward_kwargs=dict(forward_kwarg="forward_kwarg"))
        constrained_model(model_input)
        model.assert_called_once_with(model_input, forward_kwarg="forward_kwarg")

    def test_repr(self, model):
        constrained_model = modules.ConstrainedOutputModel(model, 0, forward_kwargs=dict(kwarg="kwarg"))
        assert str(constrained_model) == "ConstrainedOutputModel(model, 0, forward_kwargs={'kwarg': 'kwarg'})"
