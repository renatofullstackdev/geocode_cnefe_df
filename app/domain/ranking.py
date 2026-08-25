from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from .addressing import AddressComponent, ParsedAddress
from .models import (
    AddressCandidate,
    ComponentEvidence,
    MatchAssessment,
    RankedMatch,
    ScoreBreakdown,
)
from .parser import AddressParser
from .ranking_policy import RankingPolicy
from .vocabulary import AddressVocabulary


@dataclass(frozen=True, slots=True)
class EvidenceSummary:
    score: float = 0.0
    requested_weight: float = 0.0
    missing: tuple[str, ...] = ()
    components: tuple[ComponentEvidence, ...] = ()


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

        components = self._component_evidence(
            query,
            parsed_candidate,
        )

        retrieval_score = min(
            max(float(candidate.retrieval_score), 0.0),
            1.0,
        )

        score = self._weighted_score(
            query=query,
            text_similarity=text_similarity,
            token_coverage=token_coverage,
            retrieval_score=retrieval_score,
            component_score=components.score,
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
                    components=components.score,
                    conflict_penalty=0.0,
                ),
                component_evidence=components.components,
                missing=components.missing,
                conflicts=(),
                retrieval_strategies=candidate.retrieval_strategies,
            ),
        )

    def _weighted_score(
        self,
        *,
        query: ParsedAddress,
        text_similarity: float,
        token_coverage: float,
        retrieval_score: float,
        component_score: float,
    ) -> float:
        weights = self.policy.features

        features: list[tuple[float, float]] = [
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
        ]

        if query.components:
            features.append(
                (
                    component_score,
                    weights.components,
                )
            )

        denominator = sum(weight for _, weight in features)

        return sum(value * weight for value, weight in features) / denominator

    def _component_evidence(
        self,
        query: ParsedAddress,
        candidate: ParsedAddress,
    ) -> EvidenceSummary:
        if not query.components:
            return EvidenceSummary()

        evidence: list[ComponentEvidence] = []
        matched_weight = 0.0
        requested_weight = 0.0
        missing: list[str] = []
        used_candidates: set[int] = set()

        for requested in query.components:
            weight = self._component_weight(requested)
            requested_weight += weight

            eligible = [
                (index, candidate_component)
                for index, candidate_component in enumerate(candidate.components)
                if (index not in used_candidates and candidate_component.type == requested.type)
            ]

            best: (
                tuple[
                    int,
                    AddressComponent,
                    float,
                ]
                | None
            ) = None

            for index, candidate_component in eligible:
                score = self._value_match(
                    requested.value,
                    candidate_component.value,
                )
                score *= self._scope_factor(
                    requested,
                    candidate_component,
                )

                if best is None or score > best[2]:
                    best = (
                        index,
                        candidate_component,
                        score,
                    )

            candidate_types = tuple(dict.fromkeys(item.type for _, item in eligible))
            candidate_values = tuple(dict.fromkeys(item.value for _, item in eligible))

            if best is not None and best[2] > 0.0:
                index, best_component, best_score = best
                used_candidates.add(index)

                matched_weight += weight * best_score

                status = "exact" if best_score >= 0.99 else "partial"

                evidence.append(
                    ComponentEvidence(
                        type=requested.type,
                        requested_value=requested.value,
                        candidate_types=candidate_types,
                        candidate_values=candidate_values,
                        status=status,
                        score=best_score,
                        candidate_source=(best_component.source),
                    )
                )
                continue

            identifier = f"{requested.type} {requested.value}"
            missing.append(identifier)

            evidence.append(
                ComponentEvidence(
                    type=requested.type,
                    requested_value=requested.value,
                    candidate_types=candidate_types,
                    candidate_values=candidate_values,
                    status="missing",
                    score=0.0,
                    candidate_source=None,
                )
            )

        return EvidenceSummary(
            score=(matched_weight / requested_weight if requested_weight else 0.0),
            requested_weight=requested_weight,
            missing=tuple(missing),
            components=tuple(evidence),
        )

    def _scope_factor(
        self,
        requested: AddressComponent,
        candidate: AddressComponent,
    ) -> float:
        preference = self.vocabulary.components.get(requested.type)

        if not preference or preference.scope_preference == "any" or requested.scope == "query":
            return 1.0

        return 1.0 if candidate.scope == preference.scope_preference else 0.82

    def _value_match(
        self,
        requested: str,
        candidate: str,
    ) -> float:
        if requested == candidate:
            return 1.0

        ratio = _ratio(
            requested,
            candidate,
        )

        return ratio if ratio >= self.policy.thresholds.fuzzy_component else 0.0

    def _component_weight(
        self,
        component: AddressComponent,
    ) -> float:
        spec = self.vocabulary.components.get(component.type)

        base = spec.weight if spec else (1.45 if component.type == "GRID" else 1.0)

        return base * component.confidence

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
