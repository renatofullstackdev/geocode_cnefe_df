from __future__ import annotations

import re
from collections.abc import Iterable

from .text import SPACES_PATTERN, basic_normalize


class LocalityIndex:
    """Longest-token matching over localities loaded from the database."""

    _END = "__end__"

    def __init__(
        self,
        localities: Iterable[str],
        compact_aliases: Iterable[str] = (),
    ) -> None:
        self._trie: dict[str, dict] = {}
        aliases = sorted(set(compact_aliases), key=len, reverse=True)
        compact_pattern = (
            re.compile(
                r"(?<![A-Z0-9])(" + "|".join(re.escape(alias) for alias in aliases) + r")(?=\d)"
            )
            if aliases
            else None
        )

        def normalize_locality(raw: str) -> str:
            value = basic_normalize(raw)
            if compact_pattern:
                value = compact_pattern.sub(r"\1 ", value)
            return SPACES_PATTERN.sub(" ", value).strip()

        self.values: tuple[str, ...] = tuple(
            sorted(
                {value for raw in localities if (value := normalize_locality(raw))},
                key=lambda item: (-len(item.split()), -len(item), item),
            )
        )
        for locality in self.values:
            node = self._trie
            for token in locality.split():
                node = node.setdefault(token, {})
            node[self._END] = locality

    def find(
        self,
        tokens: list[str],
        *,
        trailing_ignored: frozenset[str] = frozenset(),
    ) -> tuple[str | None, set[int]]:
        matches: list[tuple[str, set[int]]] = []
        for start in range(len(tokens)):
            node = self._trie
            indexes: list[int] = []
            for index in range(start, len(tokens)):
                child = node.get(tokens[index])
                if child is None:
                    break
                node = child
                indexes.append(index)
                locality = node.get(self._END)
                if locality:
                    matches.append((locality, set(indexes)))
        if not matches:
            return None, set()

        exact_suffix = [match for match in matches if match[1] and max(match[1]) == len(tokens) - 1]
        if exact_suffix:
            candidates = exact_suffix
        else:
            last_meaningful = len(tokens) - 1
            while last_meaningful >= 0 and tokens[last_meaningful] in trailing_ignored:
                last_meaningful -= 1
            suffix = [match for match in matches if match[1] and max(match[1]) == last_meaningful]
            candidates = suffix or matches

        locality, indexes = max(
            candidates,
            key=lambda match: (len(match[1]), len(match[0]), min(match[1])),
        )
        return locality, indexes
