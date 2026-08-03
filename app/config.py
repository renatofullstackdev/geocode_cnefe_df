from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    database_url: str
    db_pool_min_size: int
    db_pool_max_size: int
    candidates_per_strategy: int
    max_candidate_pool: int
    candidate_cache_size: int
    vocabulary_path: Path
    ranking_policy_path: Path


def _effective_values(
    env_file: Path,
    environment: Mapping[str, str],
) -> dict[str, str]:
    file_values = {
        key: value for key, value in dotenv_values(env_file).items() if value is not None
    }
    return {**file_values, **environment}


def _positive_int(values: Mapping[str, str], name: str, default: int) -> int:
    raw = values.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


def _resolve_path(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def load_settings(
    project_root: Path | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> Settings:
    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    values = _effective_values(
        root / ".env",
        os.environ if environment is None else environment,
    )
    database_url = values.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL must be defined in .env or exported")

    min_size = _positive_int(values, "DB_POOL_MIN_SIZE", 1)
    max_size = _positive_int(values, "DB_POOL_MAX_SIZE", 10)
    if min_size > max_size:
        raise RuntimeError("DB_POOL_MIN_SIZE cannot exceed DB_POOL_MAX_SIZE")

    per_strategy = _positive_int(values, "CANDIDATES_PER_STRATEGY", 700)
    max_pool = _positive_int(values, "MAX_CANDIDATE_POOL", 3500)
    if per_strategy > max_pool:
        raise RuntimeError("CANDIDATES_PER_STRATEGY cannot exceed MAX_CANDIDATE_POOL")

    return Settings(
        project_root=root,
        database_url=database_url,
        db_pool_min_size=min_size,
        db_pool_max_size=max_size,
        candidates_per_strategy=per_strategy,
        max_candidate_pool=max_pool,
        candidate_cache_size=_positive_int(values, "CANDIDATE_CACHE_SIZE", 50_000),
        vocabulary_path=_resolve_path(
            root,
            values.get("VOCABULARY_PATH", "config/address_vocabulary.json"),
        ),
        ranking_policy_path=_resolve_path(
            root,
            values.get("RANKING_POLICY_PATH", "config/ranking_policy.json"),
        ),
    )
