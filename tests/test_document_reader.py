from pathlib import Path

import pytest

from resume_crew.document_reader import extract_text


def test_extract_text_replaces_invalid_utf8(tmp_path: Path) -> None:
    source = tmp_path / "resume.txt"
    source.write_bytes(b"Candidate \xff Name")
    assert "Candidate" in extract_text(str(source))


def test_rejects_unsupported_extension(tmp_path: Path) -> None:
    source = tmp_path / "resume.exe"
    source.write_text("not supported")
    with pytest.raises(ValueError, match="Unsupported"):
        extract_text(str(source))
