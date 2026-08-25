from __future__ import annotations

import pytest

from app.domain.models import (
    Address,
    AddressCandidate,
    Coordinates,
    MatchAssessment,
    RankedMatch,
    ScoreBreakdown,
)
from app.domain.ranking import Ranker


def candidate(
    *,
    row_id: int,
    lograd_num: str,
    complemento: str,
    cep: str,
    locality: str,
    retrieval_score: float = 0.9,
) -> AddressCandidate:
    return AddressCandidate(
        address=Address(
            row_id=row_id,
            public_id=str(row_id),
            lograd_num=lograd_num,
            complemento=complemento,
            cep=cep,
            locality=locality,
            establishment=None,
            coordinates=Coordinates(-47.8, -15.8),
        ),
        retrieval_score=retrieval_score,
        retrieval_strategies=("address_trigram",),
    )


def ranker(parser, vocabulary, ranking_policy) -> Ranker:
    return Ranker(
        parser,
        vocabulary,
        ranking_policy,
    )


def test_ranker_produces_structured_assessment(
    parser,
    vocabulary,
    ranking_policy,
):
    query = parser.parse_query("SHIS QI 15 CONJUNTO 2 CASA 1")
    item = candidate(
        row_id=1,
        lograd_num="EDF SHIS QI 15 CONJUNTO 2, SN",
        complemento="CASA 1",
        cep="71635220",
        locality="LAGO SUL",
    )
    parsed_candidate = parser.parse_record(
        lograd_num=item.address.lograd_num,
        complemento=item.address.complemento,
        locality=item.address.locality,
        establishment=item.address.establishment,
        cep=item.address.cep,
    )

    match = ranker(
        parser,
        vocabulary,
        ranking_policy,
    ).rank(
        query,
        parsed_candidate,
        item,
    )

    assert isinstance(match, RankedMatch)
    assert isinstance(match.assessment, MatchAssessment)
    assert isinstance(
        match.assessment.breakdown,
        ScoreBreakdown,
    )

    assert match.address == item.address
    breakdown = match.assessment.breakdown

    expected_score = (
        breakdown.text_similarity * ranking_policy.features.text_similarity
        + breakdown.token_coverage * ranking_policy.features.token_coverage
        + item.retrieval_score * ranking_policy.features.retrieval_score
        + breakdown.components * ranking_policy.features.components
    ) / (
        ranking_policy.features.text_similarity
        + ranking_policy.features.token_coverage
        + ranking_policy.features.retrieval_score
        + ranking_policy.features.components
    )

    assert match.assessment.score == pytest.approx(expected_score)
    assert 0.0 < match.assessment.score <= 1.0

    assert breakdown.text_similarity > 0.0
    assert breakdown.token_coverage > 0.0

    assert breakdown.locality == 0.0
    assert breakdown.road == 0.0
    assert breakdown.sector == 0.0
    assert breakdown.components == pytest.approx(1.0)
    assert breakdown.conflict_penalty == 0.0

    assert match.assessment.matched_target == "address"

    assert {
        (
            evidence.type,
            evidence.requested_value,
            evidence.status,
        )
        for evidence in match.assessment.component_evidence
    } >= {
        ("GRID", "QI 15", "exact"),
        ("CONJUNTO", "2", "exact"),
        ("CASA", "1", "exact"),
    }

    assert match.assessment.missing == ()
    assert match.assessment.conflicts == ()
    assert match.assessment.retrieval_strategies == ("address_trigram",)


def test_rank_all_returns_one_assessment_per_candidate(
    parser,
    vocabulary,
    ranking_policy,
):
    query = parser.parse_query("CASA 1")

    first = candidate(
        row_id=2,
        lograd_num="EDF QUADRA 1, SN",
        complemento="CASA 1",
        cep="70000000",
        locality="AGUAS CLARAS",
    )
    second = candidate(
        row_id=1,
        lograd_num="EDF QUADRA 2, SN",
        complemento="CASA 2",
        cep="70000001",
        locality="AGUAS CLARAS",
    )

    ranked = ranker(
        parser,
        vocabulary,
        ranking_policy,
    ).rank_all(
        query,
        [first, second],
        min_score=0.0,
        cep_relaxed=False,
    )

    assert [match.address.row_id for match in ranked] == [
        2,
        1,
    ]

    assert ranked[0].assessment.score > ranked[1].assessment.score


def test_token_coverage_is_reported_in_breakdown(
    parser,
    vocabulary,
    ranking_policy,
):
    query = parser.parse_query("RUA 4 QUADRA 10 CASA 2")
    item = candidate(
        row_id=1,
        lograd_num="EDF RUA 4 QUADRA 10, SN",
        complemento="CASA 2",
        cep="70000000",
        locality="AGUAS CLARAS",
    )

    parsed_candidate = parser.parse_record(
        lograd_num=item.address.lograd_num,
        complemento=item.address.complemento,
        locality=item.address.locality,
        establishment=item.address.establishment,
        cep=item.address.cep,
    )

    match = ranker(
        parser,
        vocabulary,
        ranking_policy,
    ).rank(
        query,
        parsed_candidate,
        item,
    )

    assert 0.0 < match.assessment.breakdown.token_coverage <= 1.0


def test_retrieval_strength_contributes_to_score(
    parser,
    vocabulary,
    ranking_policy,
):
    query = parser.parse_query("RUA 4 QUADRA 10 CASA 2")

    weak_retrieval = candidate(
        row_id=1,
        lograd_num="EDF RUA 4 QUADRA 10, SN",
        complemento="CASA 2",
        cep="70000000",
        locality="AGUAS CLARAS",
        retrieval_score=0.2,
    )
    strong_retrieval = candidate(
        row_id=1,
        lograd_num="EDF RUA 4 QUADRA 10, SN",
        complemento="CASA 2",
        cep="70000000",
        locality="AGUAS CLARAS",
        retrieval_score=0.9,
    )

    parsed_candidate = parser.parse_record(
        lograd_num=strong_retrieval.address.lograd_num,
        complemento=strong_retrieval.address.complemento,
        locality=strong_retrieval.address.locality,
        establishment=(strong_retrieval.address.establishment),
        cep=strong_retrieval.address.cep,
    )

    ranker_instance = ranker(
        parser,
        vocabulary,
        ranking_policy,
    )

    weak_match = ranker_instance.rank(
        query,
        parsed_candidate,
        weak_retrieval,
    )
    strong_match = ranker_instance.rank(
        query,
        parsed_candidate,
        strong_retrieval,
    )

    assert strong_match.assessment.score > weak_match.assessment.score


def test_text_similarity_can_match_establishment(
    parser,
    vocabulary,
    ranking_policy,
):
    query = parser.parse_query("ESCOLA CLASSE GIRASSOL")

    item = AddressCandidate(
        address=Address(
            row_id=1,
            public_id="1",
            lograd_num="EDF QUADRA 10, SN",
            complemento=None,
            cep="70000000",
            locality="AGUAS CLARAS",
            establishment="ESCOLA CLASSE GIRASSOL",
            coordinates=Coordinates(
                -47.8,
                -15.8,
            ),
        ),
        retrieval_score=0.9,
        retrieval_strategies=("establishment_trigram",),
    )

    parsed_candidate = parser.parse_record(
        lograd_num=item.address.lograd_num,
        complemento=item.address.complemento,
        locality=item.address.locality,
        establishment=item.address.establishment,
        cep=item.address.cep,
    )

    match = ranker(
        parser,
        vocabulary,
        ranking_policy,
    ).rank(
        query,
        parsed_candidate,
        item,
    )

    assert match.assessment.matched_target == ("establishment")
    assert match.assessment.breakdown.text_similarity == pytest.approx(1.0)


def test_component_score_uses_typed_structural_weights(
    parser,
    vocabulary,
    ranking_policy,
):
    query = parser.parse_query("QUADRA 10 LOTE 2")

    item = candidate(
        row_id=1,
        lograd_num="EDF LOTE 2, SN",
        complemento="",
        cep="70000000",
        locality="AGUAS CLARAS",
    )

    parsed_candidate = parser.parse_record(
        lograd_num=item.address.lograd_num,
        complemento=item.address.complemento,
        locality=item.address.locality,
        establishment=item.address.establishment,
        cep=item.address.cep,
    )

    match = ranker(
        parser,
        vocabulary,
        ranking_policy,
    ).rank(
        query,
        parsed_candidate,
        item,
    )

    quadra_weight = vocabulary.components["QUADRA"].weight
    lote_weight = vocabulary.components["LOTE"].weight

    expected_component_score = lote_weight / (quadra_weight + lote_weight)

    assert match.assessment.breakdown.components == pytest.approx(expected_component_score)

    evidence = {item.type: item for item in match.assessment.component_evidence}

    assert evidence["QUADRA"].status == "missing"
    assert evidence["LOTE"].status == "exact"

    assert match.assessment.missing == ("QUADRA 10",)
    assert match.assessment.conflicts == ()
