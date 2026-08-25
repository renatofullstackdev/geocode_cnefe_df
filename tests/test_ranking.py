from __future__ import annotations

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
    assert match.assessment.score == 0.0
    assert match.assessment.classification == "weak_candidate"
    assert match.assessment.matched_target == "address"
    assert match.assessment.component_evidence == ()
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
        1,
        2,
    ]
    assert all(match.assessment.classification == "weak_candidate" for match in ranked)
