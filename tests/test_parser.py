from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.locality import LocalityIndex
from app.domain.text import (
    basic_normalize,
    canonical_value,
    normalize_cep,
)

ADDRESS_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "address_cases.json").read_text(encoding="utf-8")
)


def test_text_normalization_removes_accents_and_punctuation():
    assert basic_normalize("  Águas-Claras, DF!  ") == "AGUAS CLARAS DF"


def test_text_normalization_handles_empty_values():
    assert basic_normalize(None) == ""
    assert basic_normalize("") == ""


def test_text_normalization_canonicalizes_numeric_values():
    assert canonical_value("001A") == "1A"
    assert canonical_value("08 E 09") == "8 E 9"


def test_text_normalization_normalizes_cep():
    assert normalize_cep("71670-260") == "71670260"
    assert normalize_cep("CEP 71670 260") == "71670260"
    assert normalize_cep("no postal code") is None


def test_longest_locality_match():
    index = LocalityIndex(
        [
            "VICENTE PIRES",
            "VICENTE PIRES TRECHO 3",
        ]
    )

    locality, indexes = index.find(
        [
            "RUA",
            "4",
            "VICENTE",
            "PIRES",
            "TRECHO",
            "3",
        ]
    )

    assert locality == "VICENTE PIRES TRECHO 3"
    assert len(indexes) == 4


@pytest.mark.parametrize(
    "case",
    ADDRESS_CASES,
    ids=lambda case: case["name"],
)
def test_parser_extracts_basic_structured_evidence(
    parser,
    case,
):
    parsed = parser.parse_query(case["query"])

    assert parsed.cep == case["cep"]
    assert parsed.locality == case["locality"]
    assert list(parsed.sectors) == case["sectors"]

    assert [(road.type, road.value) for road in parsed.roads] == [
        tuple(expected) for expected in case["roads"]
    ]

    assert [(component.type, component.value) for component in parsed.components] == [
        tuple(expected) for expected in case["components"]
    ]


def test_parser_preserves_repeated_component_occurrences(parser):
    parsed = parser.parse_query("QUADRA 10 LOTE 2 LOTE 4 AGUAS CLARAS")

    lots = [component for component in parsed.components if component.type == "LOTE"]

    assert [component.value for component in lots] == [
        "2",
        "4",
    ]
    assert [component.source for component in lots] == [
        "query",
        "query",
    ]
    assert [component.scope for component in lots] == [
        "query",
        "query",
    ]
    assert lots[0].position < lots[1].position


def test_parser_distinguishes_units_and_access_elements(parser):
    parsed = parser.parse_query("LOTE 2 APARTAMENTO 101 ENTRADA B ANDAR 3 AGUAS CLARAS")

    assert [
        (
            component.type,
            component.value,
            component.category,
        )
        for component in parsed.components
    ] == [
        ("LOTE", "2", "premise"),
        ("APARTAMENTO", "101", "unit"),
        ("ENTRADA", "B", "access"),
        ("ANDAR", "3", "access"),
    ]

    assert [
        (component.source, component.scope)
        for component in parsed.components
        if component.type in {"APARTAMENTO", "ENTRADA", "ANDAR"}
    ] == [
        ("query", "query"),
        ("query", "query"),
        ("query", "query"),
    ]


def test_parser_extracts_positional_qualifiers(parser):
    parsed = parser.parse_query("LOTE 5 FUNDOS 2 TERREO AGUAS CLARAS")

    assert [(qualifier.category, qualifier.value) for qualifier in parsed.qualifiers] == [
        ("POSITION", "FUNDOS 2"),
        ("LEVEL", "TERREO"),
    ]

    assert not any(
        component.type == "NUMERO" and component.value == "2" for component in parsed.components
    )

    assert [(qualifier.source, qualifier.scope) for qualifier in parsed.qualifiers] == [
        ("query", "query"),
        ("query", "query"),
    ]


def test_parser_supports_rural_address_components(parser):
    parsed = parser.parse_query("RODOVIA DF 001 KM 13 CHACARA 26 TAGUATINGA")

    assert {road.normalized for road in parsed.roads} >= {
        "RODOVIA DF 001",
    }

    assert {
        (component.type, component.value, component.category) for component in parsed.components
    } >= {
        ("KM", "13", "rural"),
        ("CHACARA", "26", "rural"),
    }


def test_parser_supports_condominium_hierarchies(parser):
    parsed = parser.parse_query(
        "CONDOMINIO COOPERVILLE RUA 3 QUADRA 1 RUA INTERNA 3 QUADRA 3 LOTE 6"
    )

    assert [component.value for component in parsed.components if component.type == "QUADRA"] == [
        "1",
        "3",
    ]

    assert {road.normalized for road in parsed.roads} >= {
        "RUA 3",
        "RUA INTERNA 3",
    }


def test_parser_preserves_external_and_internal_record_scopes(parser):
    parsed = parser.parse_record(
        lograd_num=("EDF QUADRA 2 CONJUNTO 2 LOTE 6 BLOCO K, SN"),
        complemento="APARTAMENTO 403",
        locality="PARANOA PARQUE",
        establishment=None,
        cep="71587-302",
    )

    lot = next(component for component in parsed.components if component.type == "LOTE")
    apartment = next(
        component for component in parsed.components if component.type == "APARTAMENTO"
    )

    assert (lot.source, lot.scope) == (
        "lograd_num",
        "external",
    )
    assert (apartment.source, apartment.scope) == (
        "complemento",
        "internal",
    )
    assert parsed.cep == "71587302"
