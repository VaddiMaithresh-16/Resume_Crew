"""Output directory management."""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

OUTPUT_ROOT = Path("output")
META_FILENAME = "run_meta.json"


def slugify(value: str, max_length: int = 48) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^\w\s-]", "", normalized).strip().lower()
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug[:max_length].strip("-") or "unknown"


def create_run_directory(candidate_name: str, job_title: str) -> tuple[Path, str]:
    """Create a fresh, uniquely-named directory under output/ for a report run."""
    # Human-readable timestamp instead of an opaque Unix epoch number.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    slug = f"{slugify(candidate_name)}__{slugify(job_title)}__{timestamp}"
    directory = OUTPUT_ROOT / slug
    suffix = 1
    while directory.exists():
        suffix += 1
        directory = OUTPUT_ROOT / f"{slug}-{suffix}"

    directory.mkdir(parents=True, exist_ok=False)
    return directory, timestamp


def write_run_meta(
    directory: Path,
    candidate: str,
    job_title: str,
    score: float,
    timestamp: str,
) -> None:
    """Write a small JSON sidecar so History/trend views don't need to re-parse Markdown."""
    meta = {
        "candidate": candidate,
        "job_title": job_title,
        "score": score,
        "timestamp": timestamp,
        "directory": directory.name,
    }
    (directory / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def list_report_runs(limit: int = 50) -> list[dict]:
    """Return metadata for past runs under output/, most recent first.

    Runs without a meta sidecar (e.g. created by an older version) are skipped
    rather than crashing the History tab.
    """
    if not OUTPUT_ROOT.is_dir():
        return []

    runs: list[dict] = []
    for directory in OUTPUT_ROOT.iterdir():
        if not directory.is_dir():
            continue
        meta_path = directory / META_FILENAME
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["path"] = str(directory)
            runs.append(meta)
        except (json.JSONDecodeError, OSError):
            continue

    runs.sort(key=lambda m: m.get("timestamp", ""), reverse=True)
    return runs[:limit]


def read_run_report(directory: str) -> dict[str, str]:
    """Read all saved Markdown report files for one past run, keyed by filename."""
    path = Path(directory)
    if not path.is_dir():
        raise NotADirectoryError(f"'{path}' is not a directory.")
    files: dict[str, str] = {}
    for md_file in sorted(path.glob("*.md")):
        files[md_file.name] = md_file.read_text(encoding="utf-8", errors="replace")
    if not files:
        raise FileNotFoundError(f"No report files found in '{path}'.")
    return files
