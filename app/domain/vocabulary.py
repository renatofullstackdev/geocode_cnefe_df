from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    canonical: str
    aliases: tuple[str, ...]
    category: str
    weight: float
    scope_preference: str


@dataclass(frozen=True, slots=True)
class AddressVocabulary:
    fingerprint: str
    noise_prefixes: frozenset[str]
    noise_phrases: tuple[str, ...]
    connectors: frozenset[str]
    stopwords: frozenset[str]
    components: dict[str, ComponentSpec]
    alias_to_component: dict[str, str]
    qualifier_aliases: dict[str, str]
    road_types: frozenset[str]
    coded_grid_prefixes: frozenset[str]
    multi_value_grid_prefixes: frozenset[str]
    sector_codes: frozenset[str]
    component_compatibility: dict[str, dict[str, float]]

    def compatibility_score(self, requested_type: str, candidate_type: str) -> float:
        if requested_type == candidate_type:
            return 1.0
        return self.component_compatibility.get(requested_type, {}).get(candidate_type, 0.0)

    @property
    def component_aliases(self) -> frozenset[str]:
        return frozenset(self.alias_to_component)

    @property
    def all_compact_aliases(self) -> frozenset[str]:
        return frozenset(
            set(self.alias_to_component) | set(self.coded_grid_prefixes) | set(self.sector_codes)
        )


def _uppercase_list(values: list[str]) -> tuple[str, ...]:
    return tuple(value.strip().upper() for value in values if value.strip())


def load_vocabulary(path: str | Path | None = None) -> AddressVocabulary:
    vocabulary_path = (
        Path(path)
        if path
        else Path(__file__).resolve().parents[2] / "config" / "address_vocabulary.json"
    )
    vocabulary_bytes = vocabulary_path.read_bytes()
    raw: dict[str, Any] = json.loads(vocabulary_bytes.decode("utf-8"))

    components: dict[str, ComponentSpec] = {}
    alias_to_component: dict[str, str] = {}
    for canonical_raw, definition in raw["component_types"].items():
        canonical = canonical_raw.upper()
        aliases = _uppercase_list(definition["aliases"])
        spec = ComponentSpec(
            canonical=canonical,
            aliases=aliases,
            category=str(definition["category"]),
            weight=float(definition["weight"]),
            scope_preference=str(definition["scope_preference"]),
        )
        components[canonical] = spec
        for alias in aliases:
            previous = alias_to_component.get(alias)
            if previous and previous != canonical:
                raise ValueError(
                    f"component alias {alias!r} maps to both {previous} and {canonical}"
                )
            alias_to_component[alias] = canonical

    qualifier_aliases: dict[str, str] = {}
    for category, aliases in raw["qualifier_types"].items():
        for alias in _uppercase_list(aliases):
            previous = qualifier_aliases.get(alias)
            if previous and previous != category:
                raise ValueError(f"qualifier alias {alias!r} has conflicting categories")
            qualifier_aliases[alias] = category

    compatibility: dict[str, dict[str, float]] = {}
    for requested_raw, candidates_raw in raw.get("component_compatibility", {}).items():
        requested = requested_raw.upper()
        if requested not in components:
            raise ValueError(f"unknown compatibility source component {requested!r}")
        candidate_scores: dict[str, float] = {}
        for candidate_raw, score_raw in candidates_raw.items():
            candidate = candidate_raw.upper()
            if candidate not in components:
                raise ValueError(f"unknown compatibility target component {candidate!r}")
            score = float(score_raw)
            if not 0.0 < score < 1.0:
                raise ValueError(
                    f"compatibility score for {requested}->{candidate} must be between 0 and 1"
                )
            candidate_scores[candidate] = score
        compatibility[requested] = candidate_scores

    return AddressVocabulary(
        fingerprint=hashlib.sha256(vocabulary_bytes).hexdigest(),
        noise_prefixes=frozenset(_uppercase_list(raw["noise_prefixes"])),
        noise_phrases=_uppercase_list(raw["noise_phrases"]),
        connectors=frozenset(_uppercase_list(raw["connectors"])),
        stopwords=frozenset(_uppercase_list(raw["stopwords"])),
        components=components,
        alias_to_component=alias_to_component,
        qualifier_aliases=qualifier_aliases,
        road_types=frozenset(_uppercase_list(raw["road_types"])),
        coded_grid_prefixes=frozenset(_uppercase_list(raw["coded_grid_prefixes"])),
        multi_value_grid_prefixes=frozenset(
            _uppercase_list(raw.get("multi_value_grid_prefixes", []))
        ),
        sector_codes=frozenset(_uppercase_list(raw["sector_codes"])),
        component_compatibility=compatibility,
    )
