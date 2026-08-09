"""Command-line interface for Resume_Crew."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from . import __version__
from .document_reader import SUPPORTED_DOCUMENT_EXTENSIONS, extract_text
from .hardware import (ollama_is_running, print_profile, resolve_ollama_profile)
from .scoring import keyword_match_score
from .storage import create_run_directory


def _disable_external_tracing() -> None:
    """Force private, quiet CrewAI execution before its package is imported."""
    os.environ["CREWAI_TRACING_ENABLED"] = "false"
    os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
    os.environ["CREWAI_DISABLE_TRACKING"] = "true"
    os.environ["OTEL_SDK_DISABLED"] = "true"


def _rank_resumes(directory: str, job_description: str, provider: str) -> None:
    # CrewAI is intentionally imported only once ranking actually runs.
    from .pipeline import build_llm, run_llm_match_score

    job_text = extract_text(job_description)
    source = Path(directory).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"'{source}' is not a directory.")

    paths = sorted(
        p for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
    )
    if not paths:
        print("No supported resume files found.")
        return

    llm, resolved_provider = build_llm(provider)
    print(f"Ranking {len(paths)} resume(s) with {resolved_provider.title()}...")

    results: list[tuple[str, float | None, str]] = []
    for index, path in enumerate(paths, 1):
        print(f"  [{index}/{len(paths)}] Scoring {path.name}...")
        try:
            resume_text = extract_text(str(path))
            score, note = run_llm_match_score(resume_text, job_text, llm)
            results.append((path.name, score, note))
        except Exception as exc:
            # BUG-07: ensure error string is never empty so the table note is always valid.
            results.append((path.name, None, str(exc) or "(unknown error)"))

    results.sort(key=lambda item: (item[1] is None, -(item[1] or 0)))

    lines = ["| Rank | Resume | Score | Notes |", "|---:|---|---:|---|"]
    for index, (name, score, note) in enumerate(results, 1):
        score_str = "--" if score is None else f"{score:.0f}%"
        safe_name = name.replace("|", "\\|")
        safe_note = (note or "").replace("|", "\\|")
        lines.append(f"| {index} | {safe_name} | {score_str} | {safe_note} |")

    report = "# Resume Ranking\n\n" + "\n".join(lines) + "\n"
    print(report)


def _run_analysis(args: argparse.Namespace) -> None:
    # CrewAI is intentionally imported only for an analysis request. Its
    # import initializes optional local storage, which ranking and hardware
    # diagnostics should never require.
    from .pipeline import build_llm, build_report_files, first_meaningful_line, run_llm_analysis

    resume_text = extract_text(args.resume)
    job_text = extract_text(args.job_description)
    candidate = first_meaningful_line(resume_text, "Candidate")
    job_title = first_meaningful_line(job_text, "Target Role")
    score = keyword_match_score(resume_text, job_text)
    llm, provider = build_llm(args.provider)

    print(f"Analyzing {candidate} against {job_title} with {provider.title()}...")

    # Pass a step callback so each of the 4 agent stages prints a progress line.
    def _on_step(msg: str) -> None:
        print(f"  {msg}")

    resume_profile, job_profile, gaps, writer_output = run_llm_analysis(
        resume_text, job_text, llm, on_step=_on_step,
    )
    files = build_report_files(
        candidate, job_title, score, resume_profile, job_profile, gaps, writer_output,
    )
    directory, _timestamp = create_run_directory(candidate, job_title)
    for name, content in files.items():
        (directory / name).write_text(content, encoding="utf-8")

    print(f"Analysis complete. Report saved to: {directory}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume_Crew — grounded, local-first resume analysis",
    )
    parser.add_argument(
        "--version", action="version", version=f"Resume_Crew {__version__}",
    )
    parser.add_argument(
        "--resume", default=os.getenv("RESUME_PATH"),
        help="Resume path (.pdf, .docx, .txt, .md)",
    )
    parser.add_argument(
        "--job-description", "--jd", dest="job_description",
        default=os.getenv("JD_PATH"), help="Job description path",
    )
    parser.add_argument(
        "--provider", choices=("auto", "ollama", "gemini"),
        default=os.getenv("LLM_PROVIDER", "auto"),
    )
    parser.add_argument(
        "--rank-resumes", metavar="DIRECTORY",
        help="Score every resume in DIRECTORY against --job-description using the LLM",
    )
    parser.add_argument(
        "--check-hardware", action="store_true",
        help="Show Ollama hardware recommendation",
    )
    parser.add_argument(
        "--ollama-profile", choices=("auto", "cuda", "mps", "cpu"),
        default=os.getenv("OLLAMA_PROFILE", "auto"),
    )
    return parser


def main() -> None:
    load_dotenv()
    _disable_external_tracing()
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.check_hardware:
            profile = resolve_ollama_profile(args.ollama_profile)
            print_profile(profile)
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            if ollama_is_running(base_url):
                print("Ollama server is reachable.")
            else:
                print("Ollama server is not reachable.")
            return

        if args.rank_resumes:
            if not args.job_description:
                parser.error("--rank-resumes requires --job-description.")
            _rank_resumes(args.rank_resumes, args.job_description, args.provider)
            return

        if not args.resume or not args.job_description:
            parser.error("--resume and --job-description are required.")
        _run_analysis(args)

    except (
        FileNotFoundError, FileExistsError, NotADirectoryError,
        OSError, RuntimeError, ValueError,
    ) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
