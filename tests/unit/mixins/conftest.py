from unittest.mock import MagicMock

import pytest


@pytest.fixture
def key():
    return MagicMock(name="key")
