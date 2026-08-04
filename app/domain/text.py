from __future__ import annotations

import re
import unicodedata

CEP_PATTERN = re.compile(r"(?<!\d)(\d{5})[-.\s]?(\d{3})(?!\d)")
TRAILING_NO_NUMBER_PATTERN = re.compile(
    r"(?:,\s*)?(?:S\.?\s*/?\s*N\.?|SEM\s+N[ÚU]MERO)\s*$",
    re.IGNORECASE,
)
NON_ALNUM_PATTERN = re.compile(r"[^A-Z0-9]+")
SPACES_PATTERN = re.compile(r"\s+")
VALUE_TOKEN_PATTERN = re.compile(r"^[A-Z0-9]+$")
NUMERIC_VALUE_PATTERN = re.compile(r"^0*(\d+)([A-Z]*)$")


def strip_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def basic_normalize(value: str | None) -> str:
    if not value:
        return ""
    upper = strip_accents(value).upper()
    return SPACES_PATTERN.sub(" ", NON_ALNUM_PATTERN.sub(" ", upper)).strip()


def canonical_value(value: str) -> str:
    value = re.sub(r"(?<=\d)E(?=\d)", " E ", value)
    value = SPACES_PATTERN.sub(" ", value).strip()
    parts: list[str] = []
    for token in value.split():
        match = NUMERIC_VALUE_PATTERN.fullmatch(token)
        if match:
            number = str(int(match.group(1) or "0"))
            parts.append(f"{number}{match.group(2)}")
        else:
            parts.append(token)
    return " ".join(parts)


def normalize_cep(value: str | None) -> str | None:
    match = CEP_PATTERN.search(value or "")
    return "".join(match.groups()) if match else None
