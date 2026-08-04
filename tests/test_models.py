from app.domain.models import (
    Address,
    Coordinates,
    MatchAssessment,
    Page,
)


def test_address_formats_available_parts():
    address = Address(
        row_id=1,
        public_id="ABC",
        lograd_num="RUA 1",
        complemento="CASA 2",
        cep="70000000",
        locality="BRASILIA",
        establishment=None,
        coordinates=Coordinates(
            longitude=-47.8,
            latitude=-15.8,
        ),
    )

    assert address.formatted == ("RUA 1, CASA 2, BRASILIA, 70000000")


def test_address_format_skips_missing_parts():
    address = Address(
        row_id=1,
        public_id=None,
        lograd_num="RUA 1",
        complemento=None,
        cep=None,
        locality="BRASILIA",
        establishment=None,
        coordinates=None,
    )

    assert address.formatted == "RUA 1, BRASILIA"


def test_page_detects_remaining_results():
    page = Page.create(
        count=10,
        total=25,
        limit=10,
        offset=10,
        truncated=False,
        total_is_exact=True,
    )

    assert page.has_more


def test_page_detects_last_result_page():
    page = Page.create(
        count=5,
        total=25,
        limit=10,
        offset=20,
        truncated=False,
        total_is_exact=True,
    )

    assert not page.has_more


def test_exact_cep_assessment_is_deterministic():
    assessment = MatchAssessment.exact_cep()

    assert assessment.score == 1.0
    assert assessment.classification == "exact_cep"
    assert assessment.matched_target == "cep"
    assert assessment.component_evidence == ()
    assert assessment.missing == ()
    assert assessment.conflicts == ()
    assert assessment.retrieval_strategies == ("exact_cep",)
