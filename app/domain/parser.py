from __future__ import annotations

import re

from .addressing import AddressComponent, ParsedAddress, RoadComponent
from .locality import LocalityIndex
from .text import (
    CEP_PATTERN,
    SPACES_PATTERN,
    VALUE_TOKEN_PATTERN,
    basic_normalize,
    canonical_value,
)
from .vocabulary import AddressVocabulary


class AddressParser:
    _CORE_COMPONENT_TYPES = frozenset(
        {
            "QUADRA",
            "CONJUNTO",
            "LOTE",
            "CASA",
            "BLOCO",
        }
    )

    def __init__(
        self,
        vocabulary: AddressVocabulary,
        locality_index: LocalityIndex,
    ) -> None:
        self.vocabulary = vocabulary
        self.locality_index = locality_index

        aliases = sorted(
            vocabulary.all_compact_aliases,
            key=len,
            reverse=True,
        )
        aliases_pattern = "|".join(re.escape(alias) for alias in aliases)
        self._label_before_digit = re.compile(rf"(?<![A-Z0-9])({aliases_pattern})(?=\d)")
        self._label_after_digit = re.compile(rf"(?<=\d)({aliases_pattern})(?=\d)")

    def normalize(self, value: str | None) -> str:
        normalized = basic_normalize(value)
        normalized = self._label_before_digit.sub(
            r"\1 ",
            normalized,
        )
        normalized = self._label_after_digit.sub(
            r" \1 ",
            normalized,
        )
        normalized = SPACES_PATTERN.sub(" ", normalized).strip()

        tokens = normalized.split()
        if tokens and tokens[0] in self.vocabulary.noise_prefixes:
            tokens = tokens[1:]
        return " ".join(tokens)

    def parse_query(self, value: str) -> ParsedAddress:
        cep_match = CEP_PATTERN.search(value)
        cep = "".join(cep_match.groups()) if cep_match else None
        without_cep = CEP_PATTERN.sub(" ", value)
        normalized = self.normalize(without_cep)
        tokens = normalized.split()
        consumed: set[int] = set()
        warnings: list[str] = []

        locality, locality_indexes = self.locality_index.find(
            tokens,
            trailing_ignored=self.vocabulary.stopwords,
        )
        consumed.update(locality_indexes)

        sectors = self._extract_sectors(
            tokens,
            locality_indexes,
            consumed,
        )
        boundary_tokens = self._boundary_tokens()
        components = self._extract_components(
            tokens,
            locality_indexes,
            boundary_tokens,
            consumed,
        )
        roads = self._extract_roads(
            tokens,
            locality_indexes,
            boundary_tokens,
            consumed,
        )

        inferred = self._infer_unlabelled_number(
            tokens,
            consumed,
            locality_indexes,
        )
        if inferred is not None:
            index, value_token = inferred
            components.append(
                AddressComponent(
                    type="NUMERO",
                    value=canonical_value(value_token),
                    raw_value=value_token,
                    source="query",
                    scope="query",
                    position=index,
                    confidence=0.65,
                    category="premise",
                )
            )
            consumed.add(index)
            warnings.append("unlabelled_number_interpreted_as_NUMERO")

        residual_tokens = tuple(
            dict.fromkeys(
                token
                for index, token in enumerate(tokens)
                if index not in consumed
                and token not in self.vocabulary.stopwords
                and token not in self.vocabulary.noise_prefixes
                and token not in self.vocabulary.alias_to_component
                and token not in self.vocabulary.qualifier_aliases
                and len(token) >= 2
            )
        )
        retrieval_tokens = self._retrieval_tokens(tokens)
        anchor = self._build_anchor(
            normalized=normalized,
            locality=locality,
            sectors=sectors,
            roads=roads,
            components=components,
            residual_tokens=residual_tokens,
        )

        return ParsedAddress(
            original=value,
            normalized=normalized,
            cep=cep,
            locality=locality,
            sectors=sectors,
            roads=tuple(sorted(roads, key=lambda item: item.position)),
            components=tuple(sorted(components, key=lambda item: item.position)),
            qualifiers=(),
            residual_tokens=residual_tokens,
            retrieval_tokens=retrieval_tokens,
            anchor=anchor,
            warnings=tuple(warnings),
        )

    def _extract_sectors(
        self,
        tokens: list[str],
        locality_indexes: set[int],
        consumed: set[int],
    ) -> tuple[str, ...]:
        sectors: list[str] = []
        for index, token in enumerate(tokens):
            if index in locality_indexes:
                continue
            if token in self.vocabulary.sector_codes:
                sectors.append(token)
                consumed.add(index)
        return tuple(dict.fromkeys(sectors))

    def _extract_components(
        self,
        tokens: list[str],
        locality_indexes: set[int],
        boundary_tokens: set[str],
        consumed: set[int],
    ) -> list[AddressComponent]:
        components: list[AddressComponent] = []

        for index in range(len(tokens) - 1):
            if index in locality_indexes:
                continue

            prefix = tokens[index]
            value_token = tokens[index + 1]

            if (
                prefix not in self.vocabulary.coded_grid_prefixes
                or not VALUE_TOKEN_PATTERN.fullmatch(value_token)
            ):
                continue

            components.append(
                AddressComponent(
                    type="GRID",
                    value=f"{prefix} {canonical_value(value_token)}",
                    raw_value=f"{prefix} {value_token}",
                    source="query",
                    scope="query",
                    position=index,
                    category="location",
                )
            )
            consumed.update({index, index + 1})

        for index, label in enumerate(tokens):
            canonical = self.vocabulary.alias_to_component.get(label)

            if canonical not in self._CORE_COMPONENT_TYPES:
                continue

            value_indexes = self._component_value_indexes(
                tokens,
                index,
                locality_indexes,
                boundary_tokens,
            )

            if not value_indexes:
                continue

            raw_value = " ".join(tokens[value_index] for value_index in value_indexes)
            spec = self.vocabulary.components[canonical]

            components.append(
                AddressComponent(
                    type=canonical,
                    value=canonical_value(raw_value),
                    raw_value=raw_value,
                    source="query",
                    scope="query",
                    position=index,
                    category=spec.category,
                )
            )

            consumed.add(index)
            consumed.update(value_indexes)

        return components

    def _component_value_indexes(
        self,
        tokens: list[str],
        start: int,
        locality_indexes: set[int],
        boundary_tokens: set[str],
    ) -> list[int]:
        indexes: list[int] = []
        cursor = start + 1

        while cursor < len(tokens) and len(indexes) < 3:
            token = tokens[cursor]
            if cursor in locality_indexes:
                break
            if token in boundary_tokens:
                break
            if not VALUE_TOKEN_PATTERN.fullmatch(token):
                break
            if token in self.vocabulary.stopwords and token not in self.vocabulary.connectors:
                break

            indexes.append(cursor)
            cursor += 1

            if (
                cursor < len(tokens)
                and tokens[cursor] in self.vocabulary.connectors
                and cursor + 1 < len(tokens)
                and VALUE_TOKEN_PATTERN.fullmatch(tokens[cursor + 1])
            ):
                indexes.extend((cursor, cursor + 1))
                break

        return indexes

    def _extract_roads(
        self,
        tokens: list[str],
        locality_indexes: set[int],
        boundary_tokens: set[str],
        consumed: set[int],
    ) -> list[RoadComponent]:
        roads: list[RoadComponent] = []

        for index, token in enumerate(tokens):
            if token not in self.vocabulary.road_types:
                continue

            value_indexes = self._road_value_indexes(
                tokens,
                index,
                boundary_tokens,
                locality_indexes,
            )

            if not value_indexes:
                continue

            roads.append(
                RoadComponent(
                    type=token,
                    value=" ".join(tokens[value_index] for value_index in value_indexes),
                    source="query",
                    scope="query",
                    position=index,
                )
            )

            consumed.add(index)
            consumed.update(value_indexes)

        return roads

    @staticmethod
    def _road_value_indexes(
        tokens: list[str],
        start: int,
        boundary_tokens: set[str],
        locality_indexes: set[int],
    ) -> list[int]:
        indexes: list[int] = []
        cursor = start + 1

        while cursor < len(tokens) and len(indexes) < 8:
            token = tokens[cursor]
            if cursor in locality_indexes or token in boundary_tokens:
                break
            if token.isdigit():
                if not indexes:
                    indexes.append(cursor)
                break
            indexes.append(cursor)
            cursor += 1

        return indexes

    def _boundary_tokens(self) -> set[str]:
        return (
            set(self.vocabulary.alias_to_component)
            | set(self.vocabulary.qualifier_aliases)
            | set(self.vocabulary.road_types)
            | set(self.vocabulary.coded_grid_prefixes)
            | set(self.vocabulary.sector_codes)
        )

    @staticmethod
    def _infer_unlabelled_number(
        tokens: list[str],
        consumed: set[int],
        locality_indexes: set[int],
    ) -> tuple[int, str] | None:
        candidates = [
            (index, token)
            for index, token in enumerate(tokens)
            if index not in consumed
            and index not in locality_indexes
            and re.fullmatch(r"\d+[A-Z]?", token)
        ]
        return candidates[-1] if candidates else None

    def _retrieval_tokens(
        self,
        tokens: list[str],
    ) -> tuple[str, ...]:
        labels = set(self.vocabulary.alias_to_component) | set(self.vocabulary.qualifier_aliases)
        return tuple(
            dict.fromkeys(
                token
                for token in tokens
                if token not in self.vocabulary.stopwords
                and token not in self.vocabulary.noise_prefixes
                and token not in labels
            )
        )

    @staticmethod
    def _build_anchor(
        *,
        normalized: str,
        locality: str | None,
        sectors: tuple[str, ...],
        roads: list[RoadComponent],
        components: list[AddressComponent],
        residual_tokens: tuple[str, ...],
    ) -> str:
        parts = [
            *(road.normalized for road in roads),
            *sectors,
            *(
                component.value
                if component.type == "GRID"
                else f"{component.type} {component.value}"
                for component in components
                if component.type in {"GRID", "QUADRA"}
            ),
            locality,
            *residual_tokens[:5],
        ]
        return " ".join(dict.fromkeys(part for part in parts if part)) or normalized
