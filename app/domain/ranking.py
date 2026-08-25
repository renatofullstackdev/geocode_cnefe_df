from __future__ import annotations

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
        score = 0.0

        return RankedMatch(
            address=candidate.address,
            assessment=MatchAssessment(
                score=score,
                classification=self._classification(score),
                matched_target="address",
                breakdown=ScoreBreakdown(
                    text_similarity=0.0,
                    token_coverage=0.0,
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

    def _classification(self, score: float) -> str:
        if score >= self.policy.thresholds.strong_candidate:
            return "strong_candidate"
        if score >= self.policy.thresholds.candidate:
            return "candidate"
        return "weak_candidate"
