from app.domain.text import (
    basic_normalize,
    canonical_value,
    normalize_cep,
)


def test_text_normalization_removes_accents_and_punctuation():
    assert (
        basic_normalize("  Águas-Claras, DF!  ")
        == "AGUAS CLARAS DF"
    )


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