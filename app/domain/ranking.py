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
    conflict_weight: float = 0.0
    missing: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
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
                match.assessment.breakdown.conflict_penalty,
                -match.assessment.breakdown.token_coverage,
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
        locality = self._locality_evidence(
            query,
            parsed_candidate,
        )
        roads = self._road_evidence(
            query,
            parsed_candidate,
        )
        sectors = self._sector_evidence(
            query,
            parsed_candidate,
        )

        summaries = (
            components,
            locality,
            roads,
            sectors,
        )

        total_conflict_weight = sum(item.conflict_weight for item in summaries)
        total_structural_weight = sum(item.requested_weight for item in summaries)

        conflict_penalty = (
            total_conflict_weight / total_structural_weight if total_structural_weight else 0.0
        )

        missing = tuple(value for item in summaries for value in item.missing)
        conflicts = tuple(value for item in summaries for value in item.conflicts)

        cep_exact = bool(query.cep and parsed_candidate.cep == query.cep)
        cep_conflict = bool(
            query.cep
            and parsed_candidate.cep
            and parsed_candidate.cep != query.cep
            and not cep_relaxed
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
            locality_score=locality.score,
            road_score=roads.score,
            sector_score=sectors.score,
            cep_exact=cep_exact,
        )

        score -= self.policy.penalties.structural_conflict * conflict_penalty

        if cep_conflict:
            score -= self.policy.penalties.cep_conflict

        if cep_relaxed:
            score -= self.policy.penalties.cep_relaxed

        if (
            query.normalized
            and query.normalized in parsed_candidate.normalized
            and conflict_penalty == 0.0
        ):
            score += self.policy.bonuses.contained_query

        score = min(max(score, 0.0), 1.0)

        classification = self._classification(
            query=query,
            score=score,
            token_coverage=token_coverage,
            component_evidence=components.components,
            conflict_penalty=conflict_penalty,
            cep_exact=cep_exact,
            cep_relaxed=cep_relaxed,
            locality_score=locality.score,
            road_score=roads.score,
            sector_score=sectors.score,
        )

        return RankedMatch(
            address=candidate.address,
            assessment=MatchAssessment(
                score=score,
                classification=classification,
                matched_target=matched_target,
                breakdown=ScoreBreakdown(
                    text_similarity=text_similarity,
                    token_coverage=token_coverage,
                    locality=locality.score,
                    road=roads.score,
                    sector=sectors.score,
                    components=components.score,
                    conflict_penalty=conflict_penalty,
                ),
                component_evidence=components.components,
                missing=missing,
                conflicts=conflicts,
                retrieval_strategies=(candidate.retrieval_strategies),
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
        locality_score: float,
        road_score: float,
        sector_score: float,
        cep_exact: bool,
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

        if query.locality:
            features.append(
                (
                    locality_score,
                    weights.locality,
                )
            )

        if query.roads:
            features.append(
                (
                    road_score,
                    weights.road,
                )
            )

        if query.sectors:
            features.append(
                (
                    sector_score,
                    weights.sector,
                )
            )

        if query.cep:
            features.append(
                (
                    1.0 if cep_exact else 0.0,
                    weights.cep,
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
        conflict_weight = 0.0
        missing: list[str] = []
        conflicts: list[str] = []
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

            if eligible:
                conflict_weight += weight

                candidate_description = ", ".join(
                    f"{item.type} {item.value}" for _, item in eligible
                )

                conflicts.append(f"{identifier} != {candidate_description}")

                evidence.append(
                    ComponentEvidence(
                        type=requested.type,
                        requested_value=requested.value,
                        candidate_types=candidate_types,
                        candidate_values=candidate_values,
                        status="conflict",
                        score=0.0,
                        candidate_source=(eligible[0][1].source),
                    )
                )
            else:
                missing.append(identifier)

                evidence.append(
                    ComponentEvidence(
                        type=requested.type,
                        requested_value=requested.value,
                        candidate_types=(),
                        candidate_values=(),
                        status="missing",
                        score=0.0,
                        candidate_source=None,
                    )
                )

        return EvidenceSummary(
            score=(matched_weight / requested_weight if requested_weight else 0.0),
            requested_weight=requested_weight,
            conflict_weight=conflict_weight,
            missing=tuple(missing),
            conflicts=tuple(conflicts),
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

    def _classification(
        self,
        *,
        query: ParsedAddress,
        score: float,
        token_coverage: float,
        component_evidence: tuple[
            ComponentEvidence,
            ...,
        ],
        conflict_penalty: float,
        cep_exact: bool,
        cep_relaxed: bool,
        locality_score: float,
        road_score: float,
        sector_score: float,
    ) -> str:
        thresholds = self.policy.thresholds

        if query.is_only_cep and cep_exact:
            return "exact_cep"

        if cep_relaxed:
            return "cep_relaxed_candidate"

        components_exact = bool(component_evidence) and all(
            item.status == "exact" for item in component_evidence
        )

        all_structured_exact = (
            components_exact
            and (not query.sectors or sector_score == 1.0)
            and (not query.locality or locality_score == 1.0)
            and (not query.roads or road_score >= thresholds.exact_road)
            and conflict_penalty == 0.0
        )

        if (
            all_structured_exact
            and token_coverage >= thresholds.exact_token_coverage
            and (not query.cep or cep_exact)
        ):
            return "exact_structured"

        if score >= thresholds.strong_candidate and conflict_penalty == 0.0:
            return "strong_candidate"

        if score >= thresholds.candidate:
            return "candidate"

        return "weak_candidate"

    def _locality_evidence(
        self,
        query: ParsedAddress,
        candidate: ParsedAddress,
    ) -> EvidenceSummary:
        if not query.locality:
            return EvidenceSummary()

        weight = self.policy.structural.locality
        identifier = f"LOCALITY {query.locality}"

        if not candidate.locality:
            return EvidenceSummary(
                requested_weight=weight,
                missing=(identifier,),
            )

        score = self._phrase_relation(
            query.locality,
            candidate.locality,
        )

        if score > 0.0:
            return EvidenceSummary(
                score=score,
                requested_weight=weight,
            )

        return EvidenceSummary(
            requested_weight=weight,
            conflict_weight=weight,
            conflicts=(f"{identifier} != {candidate.locality}",),
        )

    def _road_evidence(
        self,
        query: ParsedAddress,
        candidate: ParsedAddress,
    ) -> EvidenceSummary:
        if not query.roads:
            return EvidenceSummary()

        weight = self.policy.structural.road
        requested_weight = weight * len(query.roads)

        candidate_roads = [road.normalized for road in candidate.roads]

        if not candidate_roads:
            return EvidenceSummary(
                requested_weight=requested_weight,
                missing=tuple(f"ROAD {road.normalized}" for road in query.roads),
            )

        scores: list[float] = []
        conflicts: list[str] = []

        for requested in query.roads:
            best = max(
                _ratio(
                    requested.normalized,
                    value,
                )
                for value in candidate_roads
            )

            if best >= self.policy.thresholds.road_match:
                scores.append(best)
            else:
                scores.append(0.0)
                conflicts.append(f"ROAD {requested.normalized} != {', '.join(candidate_roads)}")

        return EvidenceSummary(
            score=sum(scores) / len(scores),
            requested_weight=requested_weight,
            conflict_weight=(weight * len(conflicts)),
            conflicts=tuple(conflicts),
        )

    def _sector_evidence(
        self,
        query: ParsedAddress,
        candidate: ParsedAddress,
    ) -> EvidenceSummary:
        if not query.sectors:
            return EvidenceSummary()

        weight = self.policy.structural.sector

        requested = tuple(dict.fromkeys(query.sectors))
        requested_weight = weight * len(requested)

        candidate_set = set(candidate.sectors)

        if not candidate_set:
            return EvidenceSummary(
                requested_weight=requested_weight,
                missing=tuple(f"SECTOR {sector}" for sector in requested),
            )

        unmatched = [sector for sector in requested if sector not in candidate_set]
        matched = len(requested) - len(unmatched)

        return EvidenceSummary(
            score=matched / len(requested),
            requested_weight=requested_weight,
            conflict_weight=(weight * len(unmatched)),
            conflicts=tuple(
                f"SECTOR {sector} != {', '.join(sorted(candidate_set))}" for sector in unmatched
            ),
        )

    def _phrase_relation(
        self,
        requested: str,
        candidate: str,
    ) -> float:
        if requested == candidate:
            return 1.0

        requested_tokens = set(requested.split())
        candidate_tokens = set(candidate.split())

        if requested_tokens and (
            requested_tokens <= candidate_tokens or candidate_tokens <= requested_tokens
        ):
            return 0.86

        ratio = _ratio(
            requested,
            candidate,
        )

        return ratio if ratio >= self.policy.thresholds.named_place else 0.0


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
