import pytest

from resume_crew.hardware import resolve_ollama_profile


def test_invalid_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="auto, cuda, mps, or cpu"):
        resolve_ollama_profile("invalid")


def test_auto_profile_returns_valid_device() -> None:
    profile = resolve_ollama_profile("auto")
    assert profile["name"] in {"cuda", "mps", "cpu"}
    assert profile["context"] > 0


def test_cpu_profile_returns_cpu() -> None:
    profile = resolve_ollama_profile("cpu")
    assert profile["name"] == "cpu"
