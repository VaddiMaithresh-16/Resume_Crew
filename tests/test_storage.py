import os
from pathlib import Path

from resume_crew.storage import create_run_directory, slugify


def test_slugify_is_filesystem_safe() -> None:
    assert slugify("A Candidate / ML & Data") == "a-candidate-ml-data"


def test_timestamp_is_human_readable(tmp_path: Path) -> None:
    """ISSUE-09: directory name must contain a readable date stamp, not a Unix epoch."""
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        _, timestamp = create_run_directory("Jane Doe", "Engineer")
        # Format must be YYYYMMDD_HHMMSS — all digits and one underscore.
        assert len(timestamp) == 15, f"Unexpected timestamp format: {timestamp!r}"
        assert timestamp[8] == "_", f"Expected underscore at position 8: {timestamp!r}"
        assert timestamp.replace("_", "").isdigit(), f"Non-digit chars in timestamp: {timestamp!r}"
    finally:
        os.chdir(original_cwd)


def test_auto_directory_name_contains_timestamp(tmp_path: Path) -> None:
    """Auto-generated directory slug should embed the human-readable timestamp."""
    # Run from tmp_path so the 'output/' folder lands inside tmp.
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        directory, timestamp = create_run_directory("Alice Smith", "Data Analyst")
        assert timestamp in directory.name, (
            f"Timestamp {timestamp!r} not found in dir name {directory.name!r}"
        )
    finally:
        os.chdir(original_cwd)


def test_repeat_run_gets_unique_directory(tmp_path: Path) -> None:
    """Two runs with the same candidate/role in the same second must not collide."""
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        first, _ = create_run_directory("Alice Smith", "Data Analyst")
        second, _ = create_run_directory("Alice Smith", "Data Analyst")
        assert first != second
        assert first.exists() and second.exists()
    finally:
        os.chdir(original_cwd)
