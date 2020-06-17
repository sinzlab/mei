from unittest.mock import MagicMock

import pytest


@pytest.fixture
def key():
    return MagicMock(name="key")


@pytest.fixture
def insert1():
    return MagicMock(name="insert1")
