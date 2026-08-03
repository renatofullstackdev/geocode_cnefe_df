from pathlib import Path

from app.config import load_settings


def test_dotenv_and_explicit_environment_are_merged_without_side_effects(
    tmp_path: Path,
    monkeypatch,
):
    (tmp_path / ".env").write_text(
        "DATABASE_URL=postgresql://from-file\nDB_POOL_MAX_SIZE=7\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DB_POOL_MAX_SIZE", "99")
    settings = load_settings(
        tmp_path,
        environment={"DATABASE_URL": "postgresql://explicit"},
    )
    assert settings.database_url == "postgresql://explicit"
    assert settings.db_pool_max_size == 7
    assert __import__("os").environ["DB_POOL_MAX_SIZE"] == "99"


def test_relative_configuration_paths_are_resolved_from_project_root(tmp_path: Path):
    (tmp_path / ".env").write_text(
        "DATABASE_URL=postgresql://db\nVOCABULARY_PATH=x.json\nRANKING_POLICY_PATH=y.json\n",
        encoding="utf-8",
    )
    settings = load_settings(tmp_path, environment={})
    assert settings.vocabulary_path == tmp_path / "x.json"
    assert settings.ranking_policy_path == tmp_path / "y.json"
