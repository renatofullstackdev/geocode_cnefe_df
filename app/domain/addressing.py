from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

SourceField = Literal["query", "lograd_num", "complemento", "locality", "establishment"]
Scope = Literal["query", "external", "internal", "metadata"]


@dataclass(frozen=True, slots=True)
class AddressComponent:
    type: str
    value: str
    raw_value: str
    source: SourceField
    scope: Scope
    position: int
    confidence: float = 1.0
    category: str = "other"


@dataclass(frozen=True, slots=True)
class RoadComponent:
    type: str
    value: str
    source: SourceField
    scope: Scope
    position: int

    @property
    def normalized(self) -> str:
        return f"{self.type} {self.value}".strip()


@dataclass(frozen=True, slots=True)
class AddressQualifier:
    category: str
    value: str
    source: SourceField
    scope: Scope
    position: int


@dataclass(frozen=True, slots=True)
class ParsedAddress:
    original: str
    normalized: str
    cep: str | None
    locality: str | None
    sectors: tuple[str, ...]
    roads: tuple[RoadComponent, ...]
    components: tuple[AddressComponent, ...]
    qualifiers: tuple[AddressQualifier, ...]
    residual_tokens: tuple[str, ...]
    retrieval_tokens: tuple[str, ...]
    anchor: str
    warnings: tuple[str, ...]
    establishment: str | None = None

    @cached_property
    def component_types(self) -> frozenset[str]:
        return frozenset(component.type for component in self.components)

    @property
    def is_only_cep(self) -> bool:
        return bool(self.cep) and not any(
            (
                self.locality,
                self.sectors,
                self.roads,
                self.components,
                self.qualifiers,
                self.residual_tokens,
                self.establishment,
            )
        )
