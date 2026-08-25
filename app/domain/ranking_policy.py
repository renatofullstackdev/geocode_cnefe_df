from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class FeatureWeights:
    text_similarity: float
    token_coverage: float
    retrieval_score: float
    components: float
    locality: float
    road: float
    sector: float
    cep: float


@dataclass(frozen=True, slots=True)
class StructuralWeights:
    locality: float
    road: float
    sector: float


@dataclass(frozen=True, slots=True)
class Penalties:
    structural_conflict: float
    cep_conflict: float
    cep_relaxed: float


@dataclass(frozen=True, slots=True)
class Bonuses:
    contained_query: float


@dataclass(frozen=True, slots=True)
class Thresholds:
    fuzzy_component: float
    named_place: float
    road_match: float
    exact_road: float
    exact_token_coverage: float
    strong_candidate: float
    candidate: float


@dataclass(frozen=True, slots=True)
class RankingPolicy:
    fingerprint: str
    features: FeatureWeights
    structural: StructuralWeights
    penalties: Penalties
    bonuses: Bonuses
    thresholds: Thresholds


def _number(mapping: dict[str, Any], name: str, *, minimum: float = 0.0) -> float:
    try:
        value = float(mapping[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid or missing ranking policy value: {name}") from exc
    if value < minimum:
        raise ValueError(f"ranking policy value {name} must be >= {minimum}")
    return value


def load_ranking_policy(path: str | Path | None = None) -> RankingPolicy:
    policy_path = (
        Path(path)
        if path
        else Path(__file__).resolve().parents[2] / "config" / "ranking_policy.json"
    )
    payload = policy_path.read_bytes()
    raw: dict[str, Any] = json.loads(payload.decode("utf-8"))

    features = raw["feature_weights"]
    structural = raw["structural_weights"]
    penalties = raw["penalties"]
    bonuses = raw["bonuses"]
    thresholds = raw["thresholds"]

    policy = RankingPolicy(
        fingerprint=hashlib.sha256(payload).hexdigest(),
        features=FeatureWeights(
            text_similarity=_number(features, "text_similarity"),
            token_coverage=_number(features, "token_coverage"),
            retrieval_score=_number(features, "retrieval_score"),
            components=_number(features, "components"),
            locality=_number(features, "locality"),
            road=_number(features, "road"),
            sector=_number(features, "sector"),
            cep=_number(features, "cep"),
        ),
        structural=StructuralWeights(
            locality=_number(structural, "locality"),
            road=_number(structural, "road"),
            sector=_number(structural, "sector"),
        ),
        penalties=Penalties(
            structural_conflict=_number(penalties, "structural_conflict"),
            cep_conflict=_number(penalties, "cep_conflict"),
            cep_relaxed=_number(penalties, "cep_relaxed"),
        ),
        bonuses=Bonuses(
            contained_query=_number(bonuses, "contained_query"),
        ),
        thresholds=Thresholds(
            fuzzy_component=_number(thresholds, "fuzzy_component"),
            named_place=_number(thresholds, "named_place"),
            road_match=_number(thresholds, "road_match"),
            exact_road=_number(thresholds, "exact_road"),
            exact_token_coverage=_number(thresholds, "exact_token_coverage"),
            strong_candidate=_number(thresholds, "strong_candidate"),
            candidate=_number(thresholds, "candidate"),
        ),
    )
    if policy.thresholds.candidate > policy.thresholds.strong_candidate:
        raise ValueError("candidate threshold cannot exceed strong_candidate threshold")
    return policy
