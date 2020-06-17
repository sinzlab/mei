from abc import ABC
from unittest.mock import MagicMock

import pytest

from mei import initial


def test_if_initial_guess_creator_is_subclass_of_abc():
    assert issubclass(initial.InitialGuessCreator, ABC)


class TestRandomNormal:
    @pytest.fixture
    def random_normal(self):
        initial.RandomNormal._create_random_tensor = MagicMock(
            name="create_random_tensor", return_value="initial_guess"
        )
        return initial.RandomNormal()

    def test_if_get_random_tensor_is_correctly_called(self, random_normal):
        random_normal(1, 2, 3)
        random_normal._create_random_tensor.assert_called_once_with(1, 2, 3)

    def test_if_initial_guess_is_returned(self, random_normal):
        assert random_normal(1, 2, 3) == "initial_guess"

    def test_repr(self, random_normal):
        assert repr(random_normal) == "RandomNormal()"
