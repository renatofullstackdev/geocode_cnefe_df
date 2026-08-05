from __future__ import annotations

import pytest

from app.domain.locality import LocalityIndex
from app.domain.parser import AddressParser
from app.domain.vocabulary import load_vocabulary


@pytest.fixture(scope="session")
def vocabulary():
    return load_vocabulary()


@pytest.fixture(scope="session")
def locality_index(vocabulary):
    localities = [
        "LAGO SUL",
        "LAGO NORTE",
        "AGUAS CLARAS",
        "ASA NORTE",
        "ASA SUL",
        "TAGUATINGA",
        "CEILANDIA",
        "VICENTE PIRES",
        "VICENTE PIRES TRECHO 3",
        "26 DE SETEMBRO",
        "RECANTO DAS EMAS",
        "ZONA INDUSTRIAL GUARA",
        "RIACHO FUNDO I",
        "RIACHO FUNDO II",
        "PARANOA PARQUE",
    ]

    return LocalityIndex(
        localities,
        vocabulary.all_compact_aliases,
    )


@pytest.fixture(scope="session")
def parser(vocabulary, locality_index):
    return AddressParser(
        vocabulary,
        locality_index,
    )
