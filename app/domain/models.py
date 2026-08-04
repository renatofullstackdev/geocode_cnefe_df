from __future__ import annotations

from dataclasses import dataclass

from .addressing import ParsedAddress


@dataclass(frozen=True, slots=True)
class Coordinates:
    longitude: float
    latitude: float


@dataclass(frozen=True, slots=True)
class Address:
    """CNEFE address plus its internal database row identifier."""

    row_id: int
    public_id: str | None
    lograd_num: str | None
    complemento: str | None
    cep: str | None
    locality: str | None
    establishment: str | None
    coordinates: Coordinates | None

    @property
    def formatted(self) -> str:
        return ", ".join(
            part for part in (self.lograd_num, self.complemento, self.locality, self.cep) if part
        )


@dataclass(frozen=True, slots=True)
class AddressCandidate:
    address: Address
    retrieval_score: float
    retrieval_strategies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ComponentEvidence:
    type: str
    requested_value: str
    candidate_types: tuple[str, ...]
    candidate_values: tuple[str, ...]
    status: str
    score: float
    candidate_source: str | None


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    text_similarity: float
    token_coverage: float
    locality: float
    road: float
    sector: float
    components: float
    conflict_penalty: float


@dataclass(frozen=True, slots=True)
class MatchAssessment:
    score: float
    classification: str
    matched_target: str
    breakdown: ScoreBreakdown
    component_evidence: tuple[ComponentEvidence, ...]
    missing: tuple[str, ...]
    conflicts: tuple[str, ...]
    retrieval_strategies: tuple[str, ...]

    @classmethod
    def exact_cep(cls) -> MatchAssessment:
        return cls(
            score=1.0,
            classification="exact_cep",
            matched_target="cep",
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
            retrieval_strategies=("exact_cep",),
        )


@dataclass(frozen=True, slots=True)
class RankedMatch:
    address: Address
    assessment: MatchAssessment


@dataclass(frozen=True, slots=True)
class Page:
    count: int
    total: int
    limit: int
    offset: int
    has_more: bool
    truncated: bool
    total_is_exact: bool

    @classmethod
    def create(
        cls,
        *,
        count: int,
        total: int,
        limit: int,
        offset: int,
        truncated: bool,
        total_is_exact: bool,
    ) -> Page:
        return cls(
            count=count,
            total=total,
            limit=limit,
            offset=offset,
            has_more=offset + count < total,
            truncated=truncated,
            total_is_exact=total_is_exact,
        )


@dataclass(frozen=True, slots=True)
class GeocodeResult:
    original_query: str
    parsed_query: ParsedAddress
    page: Page
    matches: tuple[RankedMatch, ...]
    cep_relaxed: bool


@dataclass(frozen=True, slots=True)
class ReverseMatch:
    rank: int
    address: Address
    distance_meters: float


@dataclass(frozen=True, slots=True)
class ReverseResult:
    latitude: float
    longitude: float
    radius_meters: float
    page: Page
    matches: tuple[ReverseMatch, ...]
