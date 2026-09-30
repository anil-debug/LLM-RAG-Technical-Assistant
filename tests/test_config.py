"""Configuration loads from defaults, the environment, and a dotenv file."""

from pathlib import Path

from core.settings import Settings


def test_package_imports():
    import core

    assert core.__version__ == "0.1.0"


def test_defaults_without_env_file():
    settings = Settings(_env_file=None)
    assert settings.embedding_model == "BAAI/bge-base-en-v1.5"
    assert settings.embedding_dim == 768
    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.vector_backend == "memory"
    assert settings.intel_entities is False


def test_environment_overrides_default(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "lab-model")
    settings = Settings(_env_file=None)
    assert settings.llm_model == "lab-model"


def test_dotenv_file_is_read(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=from-dotenv\nRERANKER_ENABLED=true\n", encoding="utf-8")
    settings = Settings(_env_file=env_file)
    assert settings.llm_model == "from-dotenv"
    assert settings.reranker_enabled is True


def test_environment_wins_over_dotenv(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "from-env")
    settings = Settings(_env_file=env_file)
    assert settings.llm_model == "from-env"


def test_known_model_dimension_is_checked():
    try:
        Settings(_env_file=None, embedding_model="BAAI/bge-base-en-v1.5", embedding_dim=384)
    except ValueError as exc:
        assert "768" in str(exc)
    else:
        raise AssertionError("mismatched dimension should fail")
