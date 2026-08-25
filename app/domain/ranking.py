from __future__ import annotations

from rapidfuzz import fuzz

from .addressing import ParsedAddress
from .models import (
    AddressCandidate,
    MatchAssessment,
    RankedMatch,
    ScoreBreakdown,
)
from .parser import AddressParser
from .ranking_policy import RankingPolicy
from .vocabulary import AddressVocabulary


class Ranker:
    def __init__(
        self,
        parser: AddressParser,
        vocabulary: AddressVocabulary,
        policy: RankingPolicy,
    ) -> None:
        self.parser = parser
        self.vocabulary = vocabulary
        self.policy = policy

    def rank_all(
        self,
        query: ParsedAddress,
        candidates: list[AddressCandidate],
        *,
        min_score: float,
        cep_relaxed: bool,
    ) -> list[RankedMatch]:
        ranked: list[RankedMatch] = []

        for candidate in candidates:
            address = candidate.address
            parsed_candidate = self.parser.parse_record(
                lograd_num=address.lograd_num,
                complemento=address.complemento,
                locality=address.locality,
                establishment=address.establishment,
                cep=address.cep,
            )

            match = self.rank(
                query,
                parsed_candidate,
                candidate,
                cep_relaxed=cep_relaxed,
            )

            if match.assessment.score >= min_score:
                ranked.append(match)

        ranked.sort(
            key=lambda match: (
                -match.assessment.score,
                match.address.row_id,
            )
        )
        return ranked

    def rank(
        self,
        query: ParsedAddress,
        parsed_candidate: ParsedAddress,
        candidate: AddressCandidate,
        *,
        cep_relaxed: bool = False,
    ) -> RankedMatch:
        address_similarity = _ratio(
            query.normalized,
            parsed_candidate.normalized,
        )
        establishment_similarity = _ratio(
            query.normalized,
            parsed_candidate.establishment,
        )

        if establishment_similarity > address_similarity:
            text_similarity = establishment_similarity
            matched_target = "establishment"
        else:
            text_similarity = address_similarity
            matched_target = "address"

        token_coverage = _token_coverage(
            query.retrieval_tokens,
            parsed_candidate.normalized,
        )

        retrieval_score = min(
            max(float(candidate.retrieval_score), 0.0),
            1.0,
        )

        score = self._weighted_score(
            text_similarity=text_similarity,
            token_coverage=token_coverage,
            retrieval_score=retrieval_score,
        )

        return RankedMatch(
            address=candidate.address,
            assessment=MatchAssessment(
                score=score,
                classification=self._classification(score),
                matched_target=matched_target,
                breakdown=ScoreBreakdown(
                    text_similarity=text_similarity,
                    token_coverage=token_coverage,
                    locality=0.0,
                    road=0.0,
                    sector=0.0,
                    components=0.0,
                    conflict_penalty=0.0,
                ),
                component_evidence=(),
                missing=(),
                conflicts=(),
                retrieval_strategies=candidate.retrieval_strategies,
            ),
        )

    def _weighted_score(
        self,
        *,
        text_similarity: float,
        token_coverage: float,
        retrieval_score: float,
    ) -> float:
        weights = self.policy.features

        features = (
            (
                text_similarity,
                weights.text_similarity,
            ),
            (
                token_coverage,
                weights.token_coverage,
            ),
            (
                retrieval_score,
                weights.retrieval_score,
            ),
        )

        denominator = sum(weight for _, weight in features)

        return sum(value * weight for value, weight in features) / denominator

    def _classification(self, score: float) -> str:
        if score >= self.policy.thresholds.strong_candidate:
            return "strong_candidate"
        if score >= self.policy.thresholds.candidate:
            return "candidate"
        return "weak_candidate"


def _ratio(
    left: str | None,
    right: str | None,
) -> float:
    if not left or not right:
        return 0.0

    return fuzz.WRatio(left, right) / 100.0


def _token_coverage(
    query_tokens: tuple[str, ...],
    candidate_text: str,
) -> float:
    if not query_tokens:
        return 0.0

    candidate_tokens = set(candidate_text.split())

    return sum(token in candidate_tokens for token in query_tokens) / len(query_tokens)
