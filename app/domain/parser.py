from __future__ import annotations

import re

from .addressing import (
    AddressComponent,
    AddressQualifier,
    ParsedAddress,
    RoadComponent,
    Scope,
    SourceField,
)
from .locality import LocalityIndex
from .text import (
    CEP_PATTERN,
    SPACES_PATTERN,
    TRAILING_NO_NUMBER_PATTERN,
    VALUE_TOKEN_PATTERN,
    basic_normalize,
    canonical_value,
    normalize_cep,
)
from .vocabulary import AddressVocabulary


class AddressParser:
    def __init__(self, vocabulary: AddressVocabulary, locality_index: LocalityIndex) -> None:
        self.vocabulary = vocabulary
        self.locality_index = locality_index
        aliases = sorted(vocabulary.all_compact_aliases, key=len, reverse=True)
        aliases_pattern = "|".join(re.escape(alias) for alias in aliases)
        # Split known compact labels: QI28, CASA17 and LOTE10AP102.
        self._label_before_digit = re.compile(rf"(?<![A-Z0-9])({aliases_pattern})(?=\d)")
        self._label_after_digit = re.compile(rf"(?<=\d)({aliases_pattern})(?=\d)")

    def normalize(self, value: str | None, *, remove_record_noise: bool = False) -> str:
        if not value:
            return ""
        raw = value.strip()
        if remove_record_noise:
            raw = TRAILING_NO_NUMBER_PATTERN.sub("", raw)
        normalized = basic_normalize(raw)
        normalized = self._label_before_digit.sub(r"\1 ", normalized)
        normalized = self._label_after_digit.sub(r" \1 ", normalized)
        normalized = SPACES_PATTERN.sub(" ", normalized).strip()
        tokens = normalized.split()
        if tokens and tokens[0] in self.vocabulary.noise_prefixes:
            tokens = tokens[1:]
        return " ".join(tokens)

    def parse_query(self, value: str) -> ParsedAddress:
        return self._parse_text(value, source="query", scope="query", infer_number=True)

    def parse_record(
        self,
        *,
        lograd_num: str | None,
        complemento: str | None,
        locality: str | None,
        establishment: str | None,
        cep: str | None,
    ) -> ParsedAddress:
        external = self._parse_text(
            lograd_num or "",
            source="lograd_num",
            scope="external",
            infer_number=False,
            remove_record_noise=True,
            detect_locality=False,
        )
        internal = self._parse_text(
            complemento or "",
            source="complemento",
            scope="internal",
            infer_number=False,
            remove_record_noise=True,
            detect_locality=False,
        )
        normalized_locality = self.normalize(locality, remove_record_noise=True) or None
        normalized_establishment = self.normalize(establishment, remove_record_noise=True) or None
        normalized_cep = normalize_cep(cep)

        normalized = " ".join(
            part
            for part in (
                external.normalized,
                internal.normalized,
                normalized_establishment,
                normalized_locality,
                normalized_cep,
            )
            if part
        )
        retrieval_tokens = self._retrieval_tokens(normalized.split())
        residual = tuple(dict.fromkeys((*external.residual_tokens, *internal.residual_tokens)))
        anchor = (
            " ".join(
                dict.fromkeys(
                    part
                    for part in (
                        *(road.normalized for road in external.roads),
                        *(sector for sector in external.sectors),
                        *(
                            component.value
                            for component in external.components
                            if component.type in {"GRID", "QUADRA"}
                        ),
                        normalized_locality,
                        *residual[:5],
                    )
                    if part
                )
            )
            or normalized
        )
        return ParsedAddress(
            original=" ".join(part for part in (lograd_num, complemento) if part),
            normalized=normalized,
            cep=normalized_cep,
            locality=normalized_locality,
            sectors=tuple(dict.fromkeys((*external.sectors, *internal.sectors))),
            roads=tuple((*external.roads, *internal.roads)),
            components=tuple((*external.components, *internal.components)),
            qualifiers=tuple((*external.qualifiers, *internal.qualifiers)),
            residual_tokens=residual,
            retrieval_tokens=retrieval_tokens,
            anchor=anchor,
            warnings=tuple((*external.warnings, *internal.warnings)),
            establishment=normalized_establishment,
        )

    def _parse_text(
        self,
        value: str,
        *,
        source: SourceField,
        scope: Scope,
        infer_number: bool,
        remove_record_noise: bool = False,
        detect_locality: bool = True,
    ) -> ParsedAddress:
        cep_match = CEP_PATTERN.search(value)
        cep = "".join(cep_match.groups()) if cep_match else None
        without_cep = CEP_PATTERN.sub(" ", value)
        normalized = self.normalize(without_cep, remove_record_noise=remove_record_noise)
        tokens = normalized.split()
        consumed: set[int] = set()
        warnings: list[str] = []

        locality: str | None = None
        locality_indexes: set[int] = set()
        if detect_locality:
            locality, locality_indexes = self.locality_index.find(
                tokens, trailing_ignored=self.vocabulary.stopwords
            )
            consumed.update(locality_indexes)

        sectors: list[str] = []
        for index, token in enumerate(tokens):
            if index in locality_indexes:
                continue
            if token in self.vocabulary.sector_codes:
                sectors.append(token)
                consumed.add(index)

        components: list[AddressComponent] = []
        # Coded grids are generic components and may coexist with QUADRA labels.
        for index in range(len(tokens) - 1):
            if index in locality_indexes:
                continue
            prefix = tokens[index]
            value_token = tokens[index + 1]
            if prefix in self.vocabulary.coded_grid_prefixes and VALUE_TOKEN_PATTERN.fullmatch(
                value_token
            ):
                grid_value_tokens = [value_token]
                if (
                    prefix in self.vocabulary.multi_value_grid_prefixes
                    and index + 2 < len(tokens)
                    and re.fullmatch(r"\d+[A-Z]?", tokens[index + 2])
                ):
                    grid_value_tokens.append(tokens[index + 2])
                raw_grid = " ".join((prefix, *grid_value_tokens))
                components.append(
                    AddressComponent(
                        type="GRID",
                        value=" ".join(
                            (
                                prefix,
                                *(canonical_value(token) for token in grid_value_tokens),
                            )
                        ),
                        raw_value=raw_grid,
                        source=source,
                        scope=scope,
                        position=index,
                        category="location",
                    )
                )
                consumed.update(range(index, index + 1 + len(grid_value_tokens)))

        # Some CNEFE records use postfix distance notation (for example, "65 KM").
        for index in range(len(tokens) - 1):
            if (
                re.fullmatch(r"\d+(?:[A-Z])?", tokens[index])
                and self.vocabulary.alias_to_component.get(tokens[index + 1]) == "KM"
                and (
                    index + 2 >= len(tokens)
                    or not re.fullmatch(r"\d+(?:[A-Z])?", tokens[index + 2])
                )
            ):
                spec = self.vocabulary.components["KM"]
                components.append(
                    AddressComponent(
                        type="KM",
                        value=canonical_value(tokens[index]),
                        raw_value=tokens[index],
                        source=source,
                        scope=scope,
                        position=index,
                        category=spec.category,
                    )
                )
                consumed.update({index, index + 1})

        boundary_tokens = (
            set(self.vocabulary.alias_to_component)
            | set(self.vocabulary.qualifier_aliases)
            | set(self.vocabulary.road_types)
            | set(self.vocabulary.coded_grid_prefixes)
            | set(self.vocabulary.sector_codes)
        )

        for index, label in enumerate(tokens):
            canonical = self.vocabulary.alias_to_component.get(label)

            if canonical is None:
                continue

            if canonical == "KM" and index in consumed:
                continue

            if (
                canonical == "QUADRA"
                and index + 1 < len(tokens)
                and tokens[index + 1] in self.vocabulary.coded_grid_prefixes
            ):
                consumed.add(index)
                continue
            value_indexes: list[int] = []
            cursor = index + 1
            while cursor < len(tokens) and len(value_indexes) < 6:
                token = tokens[cursor]
                if cursor in locality_indexes:
                    break
                if token in boundary_tokens and cursor not in value_indexes:
                    break
                if token in boundary_tokens:
                    break
                if not VALUE_TOKEN_PATTERN.fullmatch(token):
                    break
                if token in self.vocabulary.stopwords and token not in self.vocabulary.connectors:
                    break
                value_indexes.append(cursor)
                cursor += 1
                if cursor < len(tokens) and tokens[cursor] in self.vocabulary.connectors:
                    if cursor + 1 < len(tokens) and VALUE_TOKEN_PATTERN.fullmatch(
                        tokens[cursor + 1]
                    ):
                        value_indexes.extend((cursor, cursor + 1))
                        cursor += 2
                    else:
                        break
            if not value_indexes:
                continue
            raw_value = " ".join(tokens[value_index] for value_index in value_indexes)
            spec = self.vocabulary.components[canonical]
            components.append(
                AddressComponent(
                    type=canonical,
                    value=canonical_value(raw_value),
                    raw_value=raw_value,
                    source=source,
                    scope=scope,
                    position=index,
                    category=spec.category,
                )
            )
            consumed.add(index)
            consumed.update(value_indexes)

        qualifiers: list[AddressQualifier] = []
        for index, token in enumerate(tokens):
            category = self.vocabulary.qualifier_aliases.get(token)
            if category is None:
                continue
            value = token
            next_index = index + 1
            if (
                next_index < len(tokens)
                and next_index not in consumed
                and next_index not in locality_indexes
            ):
                next_token = tokens[next_index]
                if VALUE_TOKEN_PATTERN.fullmatch(next_token) and next_token not in boundary_tokens:
                    value = f"{token} {canonical_value(next_token)}"
                    consumed.add(next_index)
            qualifiers.append(
                AddressQualifier(
                    category=category,
                    value=value,
                    source=source,
                    scope=scope,
                    position=index,
                )
            )
            consumed.add(index)

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
            road_value = " ".join(tokens[value_index] for value_index in value_indexes)
            roads.append(
                RoadComponent(
                    type=token,
                    value=road_value,
                    source=source,
                    scope=scope,
                    position=index,
                )
            )
            consumed.add(index)
            consumed.update(value_indexes)

        if infer_number:
            inferred = self._infer_unlabelled_number(tokens, consumed, locality_indexes)
            if inferred is not None:
                index, value_token = inferred
                components.append(
                    AddressComponent(
                        type="NUMERO",
                        value=canonical_value(value_token),
                        raw_value=value_token,
                        source=source,
                        scope=scope,
                        position=index,
                        confidence=0.65,
                        category="premise",
                    )
                )
                consumed.add(index)
                warnings.append("unlabelled_number_interpreted_as_NUMERO")

        residual_tokens = tuple(
            token
            for index, token in enumerate(tokens)
            if index not in consumed
            and token not in self.vocabulary.stopwords
            and token not in self.vocabulary.noise_prefixes
            and token not in self.vocabulary.alias_to_component
            and token not in self.vocabulary.qualifier_aliases
            and len(token) >= 2
        )
        retrieval_tokens = self._retrieval_tokens(tokens)
        anchor_parts = [
            *(road.normalized for road in roads),
            *sectors,
            *(
                component.value
                if component.type == "GRID"
                else f"{component.type} {component.value}"
                for component in components
                if component.type in {"GRID", "QUADRA", "CHACARA", "ASSENTAMENTO"}
            ),
            locality,
            *residual_tokens[:5],
        ]
        anchor = " ".join(dict.fromkeys(part for part in anchor_parts if part)) or normalized

        return ParsedAddress(
            original=value,
            normalized=normalized,
            cep=cep,
            locality=locality,
            sectors=tuple(dict.fromkeys(sectors)),
            roads=tuple(sorted(roads, key=lambda item: item.position)),
            components=tuple(
                sorted(
                    self._deduplicate_components(components),
                    key=lambda item: item.position,
                )
            ),
            qualifiers=tuple(sorted(qualifiers, key=lambda item: item.position)),
            residual_tokens=tuple(dict.fromkeys(residual_tokens)),
            retrieval_tokens=retrieval_tokens,
            anchor=anchor,
            warnings=tuple(warnings),
        )

    def _road_value_indexes(
        self,
        tokens: list[str],
        start: int,
        boundary_tokens: set[str],
        locality_indexes: set[int],
    ) -> list[int]:
        road_type = tokens[start]
        indexes: list[int] = []
        cursor = start + 1
        while cursor < len(tokens) and len(indexes) < 8:
            token = tokens[cursor]
            if cursor in locality_indexes or token in boundary_tokens:
                break
            if token.isdigit():
                if road_type in {"RUA", "R", "W3"}:
                    if not indexes or (
                        indexes == [start + 1] and tokens[indexes[0]] in {"INTERNA", "INTERNO"}
                    ):
                        indexes.append(cursor)
                        cursor += 1
                        if cursor < len(tokens) and tokens[cursor] in {
                            "NORTE",
                            "SUL",
                            "LESTE",
                            "OESTE",
                        }:
                            indexes.append(cursor)
                        break
                    break
                if road_type in {"RODOVIA", "ESTRADA"}:
                    indexes.append(cursor)
                    cursor += 1
                    continue
                if road_type == "VIA" and len(indexes) <= 1:
                    indexes.append(cursor)
                    cursor += 1
                    break
                if indexes:
                    break
            indexes.append(cursor)
            cursor += 1
        return indexes

    def _infer_unlabelled_number(
        self,
        tokens: list[str],
        consumed: set[int],
        locality_indexes: set[int],
    ) -> tuple[int, str] | None:
        candidates: list[tuple[int, str]] = []
        for index, token in enumerate(tokens):
            if index in consumed or index in locality_indexes:
                continue
            if re.fullmatch(r"\d+[A-Z]?", token):
                candidates.append((index, token))
        return candidates[-1] if candidates else None

    def _retrieval_tokens(self, tokens: list[str]) -> tuple[str, ...]:
        labels = set(self.vocabulary.alias_to_component) | set(self.vocabulary.qualifier_aliases)
        return tuple(
            dict.fromkeys(
                token
                for token in tokens
                if token not in self.vocabulary.stopwords
                and token not in self.vocabulary.noise_prefixes
                and token not in labels
                and len(token) >= 1
            )
        )

    @staticmethod
    def _deduplicate_components(components: list[AddressComponent]) -> list[AddressComponent]:
        seen: set[tuple[str, str, SourceField, Scope]] = set()
        output: list[AddressComponent] = []
        for component in components:
            key = (component.type, component.value, component.source, component.scope)
            if key in seen:
                continue
            seen.add(key)
            output.append(component)
        return output
