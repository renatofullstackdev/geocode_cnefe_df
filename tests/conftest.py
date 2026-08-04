from __future__ import annotations

import pytest

from app.domain.vocabulary import load_vocabulary


@pytest.fixture(scope="session")
def vocabulary():
    return load_vocabulary()
