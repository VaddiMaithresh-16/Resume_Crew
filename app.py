"""Gradio web interface for Resume_Crew with optional ngrok live sharing."""

from __future__ import annotations

import os
import queue
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Make the package importable when running app.py directly from the project root.
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Disable CrewAI telemetry before any import of crewai touches the env.
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_DISABLE_TRACKING"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

import gradio as gr

from resume_crew.document_reader import SUPPORTED_DOCUMENT_EXTENSIONS, extract_text
from resume_crew.scoring import format_keyword_score, highlight_resume_matches, keyword_match_score
from resume_crew.storage import create_run_directory, list_report_runs, read_run_report, write_run_meta

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* ── Base — clean, professional dark theme ────────────────────────────────── */
:root {
    --bg-page:    #0b1220;
    --bg-card:    #131b2c;
    --bg-raised:  #1a2337;
    --border:     #2a3450;
    --accent:     #3b82f6;
    --accent-2:   #60a5fa;
    --text:       #e2e8f0;
    --muted:      #94a3b8;
    --radius:     8px;
}

body, .gradio-container {
    background: var(--bg-page) !important;
    color: var(--text) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

/* ── Header ─────────────────────────────────────────────────────────────── */
#app-header {
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 24px 32px;
    margin-bottom: 20px;
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
}
#app-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f8fafc !important;
    margin: 0 0 4px 0;
    letter-spacing: -0.01em;
}
#app-header p {
    color: var(--muted) !important;
    margin: 0;
    font-size: 0.9rem;
}

/* ── Cards / Panels ──────────────────────────────────────────────────────── */
.panel-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 20px !important;
    margin-bottom: 16px !important;
}

/* ── Inputs ──────────────────────────────────────────────────────────────── */
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}
.gradio-container label span {
    color: var(--muted) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
#analyze-btn {
    background: var(--accent) !important;
    border: none !important;
    color: #f8fafc !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    border-radius: 6px !important;
    padding: 11px 22px !important;
    transition: background 0.15s !important;
    cursor: pointer !important;
}
#analyze-btn:hover { background: var(--accent-2) !important; }
#rank-btn {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent-2) !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    padding: 11px 22px !important;
    transition: background 0.15s !important;
}
#rank-btn:hover { background: var(--bg-raised) !important; }

/* ── Progress box ─────────────────────────────────────────────────────────── */
#progress-box textarea {
    background: #060a13 !important;
    border: 1px solid var(--border) !important;
    color: #cbd5e1 !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.82rem !important;
    border-radius: 6px !important;
}

/* ── Score banner ─────────────────────────────────────────────────────────── */
#score-display {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-left: 4px solid var(--accent) !important;
    border-radius: var(--radius) !important;
    padding: 20px 28px !important;
}

/* ── Tabs — clean underline style, no gradient/glow ───────────────────────── */
.gradio-container .tabs {
    border: none !important;
}
.gradio-container .tab-nav {
    border-bottom: 1px solid var(--border) !important;
    gap: 4px !important;
}
.gradio-container .tab-nav button {
    background: transparent !important;
    color: var(--muted) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
    transition: color 0.15s, border-color 0.15s !important;
}
.gradio-container .tab-nav button.selected {
    background: transparent !important;
    color: var(--accent-2) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Markdown output ──────────────────────────────────────────────────────── */
.gradio-container .prose,
.gradio-container .markdown-body {
    color: var(--text) !important;
    background: transparent !important;
}
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3 {
    color: var(--accent-2) !important;
    font-weight: 700 !important;
}
.gradio-container .prose strong { color: #f1f5f9 !important; }
.gradio-container .prose code {
    background: var(--bg-raised) !important;
    color: #5eead4 !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 1px 6px !important;
}
.gradio-container .prose table {
    border-collapse: collapse !important;
    width: 100% !important;
}
.gradio-container .prose th {
    background: var(--bg-raised) !important;
    color: var(--text) !important;
    font-weight: 600 !important;
    padding: 8px 12px !important;
    border: 1px solid var(--border) !important;
}
.gradio-container .prose td {
    padding: 8px 12px !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}

/* ── Resume Highlights <mark> tags — force readable in dark mode ─────────── */
.gradio-container mark {
    background: #16653480 !important;
    color: #d1fae5 !important;
}

/* ── File upload zone ─────────────────────────────────────────────────────── */
.gradio-container .upload-container,
.gradio-container .file-preview {
    background: var(--bg-raised) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.2s !important;
}
.gradio-container .upload-container:hover {
    border-color: var(--accent) !important;
}

/* ── Scrollbars ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-page); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
"""

# ---------------------------------------------------------------------------
# ngrok setup
# ---------------------------------------------------------------------------

def _setup_ngrok(port: int = 7860) -> str | None:
    """Connect an ngrok tunnel if NGROK_AUTHTOKEN is present in the environment."""
    token = os.getenv("NGROK_AUTHTOKEN", "").strip()
    if not token:
        return None
    try:
        from pyngrok import conf, ngrok
        conf.get_default().auth_token = token
        tunnel = ngrok.connect(port, "http")
        url = tunnel.public_url
        print(f"\n{'='*60}")
        print(f"  🌐  Live ngrok URL: {url}")
        print(f"{'='*60}\n")
        return url
    except ImportError:
        print("[app] pyngrok is not installed. Run: pip install pyngrok")
        return None
    except Exception as exc:
        print(f"[app] ngrok setup failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Analysis worker (runs on a background thread to avoid blocking Gradio)
# ---------------------------------------------------------------------------

def _run_analysis_thread(
    resume_path: str,
    job_path: str,
    provider: str,
    step_q: "queue.Queue[str | None]",
    result_q: "queue.Queue[tuple]",
    cancel_event: "threading.Event",
) -> None:
    """Execute the full pipeline in a worker thread and push results to queues."""
    try:
        from resume_crew.pipeline import (
            AnalysisCancelled, build_llm, build_report_files,
            first_meaningful_line, run_llm_analysis,
        )

        step_q.put("📄 Reading documents...")
        resume_text = extract_text(resume_path)
        job_text = extract_text(job_path)

        step_q.put("🔢 Computing keyword match score...")
        score = keyword_match_score(resume_text, job_text)

        candidate = first_meaningful_line(resume_text, "Candidate")
        job_title = first_meaningful_line(job_text, "Target Role")

        step_q.put(f"🤖 Connecting to {provider.title()} LLM...")
        llm, resolved_provider = build_llm(provider)
        step_q.put(f"✅ Connected via {resolved_provider.title()}. Starting 4-step analysis...")

        def on_step(msg: str) -> None:
            if cancel_event.is_set():
                raise AnalysisCancelled("Analysis cancelled by user.")
            step_q.put(f"  {msg}")

        resume_profile, job_profile, gap_analysis, writer_output = run_llm_analysis(
            resume_text, job_text, llm, on_step=on_step,
        )
        if cancel_event.is_set():
            raise AnalysisCancelled("Analysis cancelled by user.")

        step_q.put("💾 Assembling report files...")
        files = build_report_files(
            candidate, job_title, score,
            resume_profile, job_profile, gap_analysis, writer_output,
        )
        directory, timestamp = create_run_directory(candidate, job_title)
        for name, content in files.items():
            (directory / name).write_text(content, encoding="utf-8")
        write_run_meta(directory, candidate, job_title, score.score, timestamp)

        step_q.put(f"✅ Done! Report saved to: {directory}")
        result_q.put(("ok", score, resume_profile, job_profile, gap_analysis,
                      writer_output, str(directory), files, resume_text, candidate, job_title))

    except AnalysisCancelled as exc:
        step_q.put(f"🛑 {exc}")
        result_q.put(("cancelled", str(exc)))
    except Exception as exc:  # noqa: BLE001
        step_q.put(f"❌ Error: {exc}")
        result_q.put(("error", str(exc)))
    finally:
        step_q.put(None)  # Sentinel — signals the generator to stop polling.


# ---------------------------------------------------------------------------
# Gradio event handlers
# ---------------------------------------------------------------------------

EMPTY_TABS = ("", "", "", "", "", "", "", "", None, None)  # score..interview, report, highlight, pdf, docx


def _get_file_path(file_obj) -> str:
    """Extract a real filesystem path from a Gradio file object (Gradio 4 or 5)."""
    if file_obj is None:
        return ""
    # Gradio 4: NamedString / TemporaryFileWrapper with .name attribute.
    if hasattr(file_obj, "name"):
        return str(file_obj.name)
    # Gradio 5+: plain string path.
    return str(file_obj)


def analyze_stream(resume_file, jd_file, provider, cancel_event_state):
    """Generator: streams progress to the UI while a background thread runs the pipeline."""
    resume_path = _get_file_path(resume_file)
    jd_path = _get_file_path(jd_file)
    if not resume_path or not jd_path:
        yield "⚠️ Please upload both a resume and a job description.", *EMPTY_TABS, cancel_event_state
        return

    cancel_event = threading.Event()
    step_q: queue.Queue[str | None] = queue.Queue()
    result_q: queue.Queue[tuple] = queue.Queue()

    thread = threading.Thread(
        target=_run_analysis_thread,
        args=(resume_path, jd_path, provider, step_q, result_q, cancel_event),
        daemon=True,
    )
    thread.start()

    log_lines: list[str] = []

    # Stream progress messages until the thread signals completion. The
    # cancel event is yielded on every tick so the Cancel button always has
    # a live reference to THIS run, even if a previous run's event is stale.
    while True:
        try:
            msg = step_q.get(timeout=1.0)
            if msg is None:
                break
            log_lines.append(msg)
            yield "\n".join(log_lines), *EMPTY_TABS, cancel_event
        except queue.Empty:
            # No new message yet — yield a heartbeat dot to show we're alive.
            yield "\n".join(log_lines) + "\n⏳ Working...", *EMPTY_TABS, cancel_event

    # Retrieve final result and populate all output tabs.
    result = result_q.get()
    if result[0] == "cancelled":
        yield f"🛑 {result[1]}", *EMPTY_TABS, None
        return
    if result[0] == "error":
        yield f"❌ Analysis failed:\n\n{result[1]}", *EMPTY_TABS, None
        return

    (_, score, resume_profile, job_profile, gap_analysis,
     writer_output, report_dir, files, resume_text, candidate, job_title) = result

    # Build the per-tab content.
    score_md = format_keyword_score(score)
    score_pct = score.score

    # Construct a visual score gauge using Unicode blocks.
    filled = int(score_pct / 5)       # 0–20 blocks
    bar = "█" * filled + "░" * (20 - filled)
    color_label = (
        "🟢 Strong" if score_pct >= 65
        else "🟡 Moderate" if score_pct >= 35
        else "🔴 Low"
    )
    score_banner = (
        f"## 📊 Keyword Match Score\n\n"
        f"### `{score_pct}%` — {color_label}\n\n"
        f"`{bar}` {score_pct}/100\n\n"
        f"{score_md}\n\n"
        f"---\n*Report saved to `{report_dir}`*"
    )

    bullets, interview = split_writer_output_safe(writer_output)
    match_report = files["match_report.md"]
    highlight_html = highlight_resume_matches(resume_text, score)

    # Export PDF (default/primary download) and Word (secondary) versions
    # of the combined report. Each export is its own try/except so one
    # format failing (e.g. a missing optional dependency) doesn't take the
    # other down with it, and neither blocks the rest of the report.
    pdf_path = None
    docx_path = None
    from resume_crew.pipeline import build_docx_report, build_pdf_report

    try:
        pdf_bytes = build_pdf_report(files, candidate, job_title)
        pdf_path = str(Path(report_dir) / "match_report.pdf")
        Path(pdf_path).write_bytes(pdf_bytes)
    except Exception as exc:  # noqa: BLE001
        log_lines.append(f"⚠️ PDF export failed: {exc}")

    try:
        docx_bytes = build_docx_report(files, candidate, job_title)
        docx_path = str(Path(report_dir) / "match_report.docx")
        Path(docx_path).write_bytes(docx_bytes)
    except Exception as exc:  # noqa: BLE001
        log_lines.append(f"⚠️ Word export failed: {exc}")

    yield (
        "\n".join(log_lines),   # progress log
        score_banner,           # score tab
        resume_profile,         # resume profile tab
        job_profile,            # job profile tab
        gap_analysis,           # gap analysis tab
        bullets,                # resume bullets tab
        interview,              # interview prep tab
        match_report,           # full match report tab
        highlight_html,         # resume with matched keywords highlighted
        pdf_path,               # downloadable .pdf report (default)
        docx_path,              # downloadable .docx report (secondary)
        None,                   # clear cancel_event state — run is done
    )


def split_writer_output_safe(writer_output: str) -> tuple[str, str]:
    try:
        from resume_crew.pipeline import split_writer_output
        return split_writer_output(writer_output)
    except ValueError:
        return writer_output, ""


def cancel_analysis(cancel_event_state):
    """Signal the running analysis thread to stop at its next checkpoint."""
    if cancel_event_state is not None:
        cancel_event_state.set()
    return "🛑 Cancelling — this finishes after the current step..."


def rank_resumes_fn(directory_path: str, jd_file, provider: str):
    """Rank all resumes in a folder against a job description using the LLM."""
    if not directory_path or not directory_path.strip():
        yield "⚠️ Please enter a folder path containing resume files."
        return
    if jd_file is None:
        yield "⚠️ Please upload a job description file."
        return

    try:
        jd_path = _get_file_path(jd_file)
        job_text = extract_text(jd_path)
        source = Path(directory_path.strip()).expanduser().resolve()
        if not source.is_dir():
            yield f"❌ `{source}` is not a directory."
            return

        paths = sorted(
            p for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
        )
        if not paths:
            yield "No supported resume files found in that directory."
            return

        from resume_crew.pipeline import build_llm, run_llm_match_score

        yield f"🤖 Connecting to {provider.title()} LLM..."
        llm, resolved_provider = build_llm(provider)

        results: list[tuple[str, float | None, str]] = []
        for idx, path in enumerate(paths, 1):
            yield (
                f"✅ Connected via {resolved_provider.title()}.\n"
                f"📄 Scoring {idx}/{len(paths)}: {path.name}..."
            )
            try:
                resume_text = extract_text(str(path))
                score, note = run_llm_match_score(resume_text, job_text, llm)
                results.append((path.name, score, note))
            except Exception as exc:
                results.append((path.name, None, str(exc) or "(unknown error)"))

        results.sort(key=lambda item: (item[1] is None, -(item[1] or 0)))

        lines = ["| Rank | Resume | Score | Notes |", "|---:|---|---:|---|"]
        for idx, (name, score, note) in enumerate(results, 1):
            score_str = "--" if score is None else f"{score:.0f}%"
            safe_note = (note or "").replace("|", chr(92) + "|")
            lines.append(f"| {idx} | {name.replace('|', chr(92)+'|')} | {score_str} | {safe_note} |")

        yield "# Resume Ranking\n\n" + "\n".join(lines)
    except Exception as exc:
        yield f"❌ {exc}"


def batch_analyze_fn(directory_path: str, jd_file, provider: str):
    """Run the FULL 4-step analysis for every resume in a folder against one
    job description, saving a separate report per resume (unlike Rank Resumes,
    which only computes a lightweight score)."""
    if not directory_path or not directory_path.strip():
        yield "⚠️ Please enter a folder path containing resume files."
        return
    if jd_file is None:
        yield "⚠️ Please upload a job description file."
        return

    try:
        jd_path = _get_file_path(jd_file)
        job_text = extract_text(jd_path)
        source = Path(directory_path.strip()).expanduser().resolve()
        if not source.is_dir():
            yield f"❌ `{source}` is not a directory."
            return

        paths = sorted(
            p for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
        )
        if not paths:
            yield "No supported resume files found in that directory."
            return

        from resume_crew.pipeline import (
            build_llm, build_report_files, first_meaningful_line, run_llm_analysis,
        )

        yield f"🤖 Connecting to {provider.title()} LLM..."
        llm, resolved_provider = build_llm(provider)
        job_title = first_meaningful_line(job_text, "Target Role")

        results: list[tuple[str, float | None, str]] = []
        log_lines = [f"✅ Connected via {resolved_provider.title()}."]
        for idx, path in enumerate(paths, 1):
            log_lines.append(f"📄 Full analysis {idx}/{len(paths)}: {path.name}...")
            yield "\n".join(log_lines)
            try:
                resume_text = extract_text(str(path))
                score = keyword_match_score(resume_text, job_text)
                candidate = first_meaningful_line(resume_text, path.stem)

                resume_profile, job_profile, gap_analysis, writer_output = run_llm_analysis(
                    resume_text, job_text, llm,
                )
                files = build_report_files(
                    candidate, job_title, score,
                    resume_profile, job_profile, gap_analysis, writer_output,
                )
                directory, timestamp = create_run_directory(candidate, job_title)
                for name, content in files.items():
                    (directory / name).write_text(content, encoding="utf-8")
                write_run_meta(directory, candidate, job_title, score.score, timestamp)

                results.append((path.name, score.score, str(directory)))
            except Exception as exc:  # noqa: BLE001
                results.append((path.name, None, f"❌ {exc}"))

        results.sort(key=lambda item: (item[1] is None, -(item[1] or 0)))
        lines = ["| Rank | Resume | Score | Saved To |", "|---:|---|---:|---|"]
        for idx, (name, score_val, note) in enumerate(results, 1):
            score_str = "--" if score_val is None else f"{score_val:.0f}%"
            safe_name = name.replace("|", chr(92) + "|")
            safe_note = note.replace("|", chr(92) + "|")
            lines.append(f"| {idx} | {safe_name} | {score_str} | `{safe_note}` |")

        log_lines.append("✅ Batch analysis complete.")
        yield "\n".join(log_lines) + "\n\n# Batch Analysis Results\n\n" + "\n".join(lines)
    except Exception as exc:
        yield f"❌ {exc}"


def compare_jds_fn(resume_file, jd_files, provider: str):
    """Score one resume against several job descriptions to see which fits best."""
    resume_path = _get_file_path(resume_file)
    if not resume_path:
        yield "⚠️ Please upload a resume."
        return
    if not jd_files:
        yield "⚠️ Please upload one or more job description files."
        return

    try:
        resume_text = extract_text(resume_path)
        from resume_crew.pipeline import build_llm, run_llm_match_score

        yield f"🤖 Connecting to {provider.title()} LLM..."
        llm, resolved_provider = build_llm(provider)

        jd_paths = [_get_file_path(f) for f in jd_files]
        results: list[tuple[str, float | None, str]] = []
        for idx, jd_path in enumerate(jd_paths, 1):
            name = Path(jd_path).name
            yield (
                f"✅ Connected via {resolved_provider.title()}.\n"
                f"📄 Scoring {idx}/{len(jd_paths)}: {name}..."
            )
            try:
                job_text = extract_text(jd_path)
                score, note = run_llm_match_score(resume_text, job_text, llm)
                results.append((name, score, note))
            except Exception as exc:
                results.append((name, None, str(exc) or "(unknown error)"))

        results.sort(key=lambda item: (item[1] is None, -(item[1] or 0)))
        lines = ["| Rank | Job Description | Score | Notes |", "|---:|---|---:|---|"]
        for idx, (name, score, note) in enumerate(results, 1):
            score_str = "--" if score is None else f"{score:.0f}%"
            safe_note = (note or "").replace("|", chr(92) + "|")
            lines.append(f"| {idx} | {name.replace('|', chr(92)+'|')} | {score_str} | {safe_note} |")

        yield "# Job Description Comparison\n\n" + "\n".join(lines)
    except Exception as exc:
        yield f"❌ {exc}"


def refresh_history_fn():
    """List past runs, most recent first, with a score-trend line chart."""
    runs = list_report_runs()
    if not runs:
        return gr.Dropdown(choices=[], value=None), "No past runs found yet — analyze a resume first.", None

    choices = [
        (f"{r.get('timestamp', '?')} — {r.get('candidate', '?')} vs {r.get('job_title', '?')} "
         f"({r.get('score', 0):.0f}%)", r["path"])
        for r in runs
    ]

    fig = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        chrono = list(reversed(runs))  # oldest → newest, left to right
        x = list(range(1, len(chrono) + 1))
        y = [r.get("score", 0) for r in chrono]
        labels = [str(r.get("candidate", "?"))[:12] for r in chrono]

        # Match the app's dark theme instead of matplotlib's default white
        # figure — an unstyled chart stands out badly against a dark UI.
        bg, card, grid, text, accent = "#0b1220", "#131b2c", "#2a3450", "#e2e8f0", "#60a5fa"
        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(card)
        ax.plot(x, y, marker="o", color=accent, linewidth=2)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Keyword match score (%)", color=text)
        ax.set_title("Score trend across past runs", color=text)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8, color=text)
        ax.tick_params(axis="y", colors=text)
        ax.grid(color=grid, linewidth=0.6, alpha=0.6)
        for spine in ax.spines.values():
            spine.set_color(grid)
        fig.tight_layout()
    except Exception:
        fig = None

    return gr.Dropdown(choices=choices, value=choices[0][1]), f"Found {len(runs)} past run(s).", fig


def load_history_run_fn(selected_path: str):
    """Load a previously saved report back into the History tab's Markdown panes."""
    if not selected_path:
        return "", "", "", "", ""
    try:
        files = read_run_report(selected_path)
        return (
            files.get("skills_gap_analysis.md", "_(not found)_"),
            files.get("resume_profile.md", "_(not found)_"),
            files.get("job_description_profile.md", "_(not found)_"),
            files.get("tailored_resume_bullets.md", "_(not found)_"),
            files.get("interview_preparation.md", "_(not found)_"),
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        return f"⚠️ {exc}", "", "", "", ""


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

PROVIDER_CHOICES = ["auto", "gemini", "ollama"]
# Single source of truth — stays in sync with document_reader.py automatically.
ACCEPTED_TYPES = list(SUPPORTED_DOCUMENT_EXTENSIONS)

# Gradio 6 moved `theme`/`css` from the Blocks() constructor to launch();
# passing them to Blocks() there is silently dropped (only a console warning),
# which would make the whole custom dark theme vanish on a fresh install.
# Detect the installed major version and route the params to wherever that
# version actually applies them, so styling never silently disappears.
_GRADIO_MAJOR = int(gr.__version__.split(".")[0]) if gr.__version__[:1].isdigit() else 4
_BLOCKS_STYLE_KWARGS: dict = {}
_LAUNCH_STYLE_KWARGS: dict = {}
if _GRADIO_MAJOR >= 6:
    _LAUNCH_STYLE_KWARGS = {"theme": gr.themes.Base(), "css": CUSTOM_CSS}
else:
    _BLOCKS_STYLE_KWARGS = {"theme": gr.themes.Base(), "css": CUSTOM_CSS}

with gr.Blocks(
    title="Resume_Crew",
    analytics_enabled=False,
    **_BLOCKS_STYLE_KWARGS,
) as demo:

    # ── Header ────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="app-header">
        <h1>🎯 Resume_Crew</h1>
        <p>Grounded, evidence-based resume &amp; job description analysis powered by AI</p>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Analyze ─────────────────────────────────────────────────
        with gr.Tab("✨ Analyze Resume"):

            with gr.Row():
                with gr.Column(scale=1):
                    resume_input = gr.File(
                        label="Resume",
                        file_types=ACCEPTED_TYPES,
                        elem_id="resume-upload",
                    )
                with gr.Column(scale=1):
                    jd_input = gr.File(
                        label="Job Description",
                        file_types=ACCEPTED_TYPES,
                        elem_id="jd-upload",
                    )

            with gr.Row():
                provider_dd = gr.Dropdown(
                    choices=PROVIDER_CHOICES,
                    value=os.getenv("LLM_PROVIDER", "auto"),
                    label="LLM Provider",
                    info="'auto' uses local Ollama if running, then Gemini",
                    scale=1,
                )
                analyze_btn = gr.Button(
                    "🚀 Analyze",
                    elem_id="analyze-btn",
                    scale=2,
                    variant="primary",
                )
                cancel_btn = gr.Button(
                    "🛑 Cancel",
                    scale=1,
                    variant="stop",
                )

            cancel_event_state = gr.State(None)

            progress_box = gr.Textbox(
                label="Progress",
                lines=5,
                max_lines=12,
                interactive=False,
                elem_id="progress-box",
                placeholder="Progress will appear here once analysis starts...",
            )

            # ── Results ───────────────────────────────────────────────────
            with gr.Tabs():
                with gr.Tab("📊 Score"):
                    score_out = gr.Markdown(elem_id="score-display")
                with gr.Tab("📝 Resume Profile"):
                    resume_profile_out = gr.Markdown()
                with gr.Tab("💼 Job Profile"):
                    job_profile_out = gr.Markdown()
                with gr.Tab("🔍 Gap Analysis"):
                    gap_out = gr.Markdown()
                with gr.Tab("✏️ Resume Bullets"):
                    bullets_out = gr.Markdown()
                with gr.Tab("🎤 Interview Prep"):
                    interview_out = gr.Markdown()
                with gr.Tab("🖍️ Resume Highlights"):
                    gr.Markdown(
                        "Matched keywords highlighted directly in your resume text. "
                        "Missing keywords can't be highlighted here since they don't appear "
                        "in the resume — see the Score tab for that list."
                    )
                    highlight_out = gr.HTML()
                with gr.Tab("📄 Full Report"):
                    report_out = gr.Markdown()
                    pdf_file_out = gr.File(label="⬇️ Download Full Report (PDF)", interactive=False)
                    docx_file_out = gr.File(label="⬇️ Download as Word (.docx)", interactive=False)

            analyze_outputs = [
                progress_box,
                score_out,
                resume_profile_out,
                job_profile_out,
                gap_out,
                bullets_out,
                interview_out,
                report_out,
                highlight_out,
                pdf_file_out,
                docx_file_out,
                cancel_event_state,
            ]
            # Every yield in analyze_stream() must produce exactly this many
            # values (1 progress string + EMPTY_TABS + 1 cancel-event slot).
            # This mismatch has been the single most common regression when
            # extending the Analyze tab — catch it at startup, not at click time.
            _expected_output_count = 1 + len(EMPTY_TABS) + 1
            assert len(analyze_outputs) == _expected_output_count, (
                f"analyze_outputs has {len(analyze_outputs)} components but "
                f"analyze_stream() yields {_expected_output_count} values — "
                "update EMPTY_TABS and every yield in analyze_stream to match."
            )

            analyze_btn.click(
                fn=analyze_stream,
                inputs=[resume_input, jd_input, provider_dd, cancel_event_state],
                outputs=analyze_outputs,
            )
            cancel_btn.click(
                fn=cancel_analysis,
                inputs=[cancel_event_state],
                outputs=[progress_box],
            )

        # ── Tab 2: Rank Resumes ────────────────────────────────────────────
        with gr.Tab("📈 Rank Resumes"):

            gr.Markdown(
                "Score every resume in a folder against a job description **using the LLM**. "
                "Each resume gets its own scoring call, so this can take a while for large folders."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    dir_input = gr.Textbox(
                        label="Resume Folder Path",
                        placeholder="e.g. C:\\Users\\you\\Resumes  or  ./samples/Resumes",
                        elem_id="dir-input",
                    )
                with gr.Column(scale=1):
                    rank_jd_input = gr.File(
                        label="Job Description",
                        file_types=ACCEPTED_TYPES,
                    )

            rank_provider_dd = gr.Dropdown(
                choices=PROVIDER_CHOICES,
                value=os.getenv("LLM_PROVIDER", "auto"),
                label="LLM Provider",
                info="'auto' uses local Ollama if running, then Gemini",
            )

            rank_btn = gr.Button("📊 Rank Resumes", elem_id="rank-btn", variant="secondary")
            rank_out = gr.Markdown(label="Ranking Results")

            rank_btn.click(
                fn=rank_resumes_fn,
                inputs=[dir_input, rank_jd_input, rank_provider_dd],
                outputs=rank_out,
            )

        # ── Tab: Batch Analyze ──────────────────────────────────────────────
        with gr.Tab("📦 Batch Analyze"):

            gr.Markdown(
                "Run the **full 4-step analysis** (profiles, gap analysis, bullets, interview "
                "prep) for every resume in a folder against one job description — a separate "
                "saved report per resume. Slower than Rank Resumes since every resume gets the "
                "complete pipeline, not just a quick score."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    batch_dir_input = gr.Textbox(
                        label="Resume Folder Path",
                        placeholder="e.g. C:\\Users\\you\\Resumes  or  ./samples/Resumes",
                    )
                with gr.Column(scale=1):
                    batch_jd_input = gr.File(
                        label="Job Description",
                        file_types=ACCEPTED_TYPES,
                    )

            batch_provider_dd = gr.Dropdown(
                choices=PROVIDER_CHOICES,
                value=os.getenv("LLM_PROVIDER", "auto"),
                label="LLM Provider",
                info="'auto' uses local Ollama if running, then Gemini",
            )

            batch_btn = gr.Button("📦 Batch Analyze", variant="secondary")
            batch_out = gr.Markdown(label="Batch Results")

            batch_btn.click(
                fn=batch_analyze_fn,
                inputs=[batch_dir_input, batch_jd_input, batch_provider_dd],
                outputs=batch_out,
            )

        # ── Tab: Compare Job Descriptions ──────────────────────────────────
        with gr.Tab("📑 Compare JDs"):

            gr.Markdown(
                "Score **one resume** against **several job descriptions** at once, to see "
                "which posting it fits best — the mirror image of Rank Resumes."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    compare_resume_input = gr.File(
                        label="Resume",
                        file_types=ACCEPTED_TYPES,
                    )
                with gr.Column(scale=2):
                    compare_jd_inputs = gr.File(
                        label="Job Descriptions (select multiple)",
                        file_types=ACCEPTED_TYPES,
                        file_count="multiple",
                    )

            compare_provider_dd = gr.Dropdown(
                choices=PROVIDER_CHOICES,
                value=os.getenv("LLM_PROVIDER", "auto"),
                label="LLM Provider",
                info="'auto' uses local Ollama if running, then Gemini",
            )

            compare_btn = gr.Button("📑 Compare", variant="secondary")
            compare_out = gr.Markdown(label="Comparison Results")

            compare_btn.click(
                fn=compare_jds_fn,
                inputs=[compare_resume_input, compare_jd_inputs, compare_provider_dd],
                outputs=compare_out,
            )

        # ── Tab: History ────────────────────────────────────────────────────
        with gr.Tab("🕘 History"):

            gr.Markdown("Browse past analysis runs saved under `output/` and see your score trend.")

            history_refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
            history_status = gr.Markdown()
            history_trend_plot = gr.Plot(label="Score trend")
            history_run_dd = gr.Dropdown(label="Past runs", choices=[])

            with gr.Tabs():
                with gr.Tab("🔍 Gap Analysis"):
                    history_gap_out = gr.Markdown()
                with gr.Tab("📝 Resume Profile"):
                    history_resume_out = gr.Markdown()
                with gr.Tab("💼 Job Profile"):
                    history_job_out = gr.Markdown()
                with gr.Tab("✏️ Resume Bullets"):
                    history_bullets_out = gr.Markdown()
                with gr.Tab("🎤 Interview Prep"):
                    history_interview_out = gr.Markdown()

            history_refresh_btn.click(
                fn=refresh_history_fn,
                inputs=[],
                outputs=[history_run_dd, history_status, history_trend_plot],
            )
            history_run_dd.change(
                fn=load_history_run_fn,
                inputs=[history_run_dd],
                outputs=[
                    history_gap_out, history_resume_out, history_job_out,
                    history_bullets_out, history_interview_out,
                ],
            )
            demo.load(
                fn=refresh_history_fn,
                inputs=[],
                outputs=[history_run_dd, history_status, history_trend_plot],
            )

        # ── Tab 3: Hardware Info ───────────────────────────────────────────
        with gr.Tab("🖥️ Hardware"):

            gr.Markdown("Detects your GPU, memory, and recommends an Ollama compute profile.")

            hw_btn = gr.Button("🔍 Detect Hardware", variant="secondary")
            hw_out = gr.Markdown()

            def _check_hw():
                from resume_crew.hardware import (
                    ollama_is_running, resolve_ollama_profile,
                )
                profile = resolve_ollama_profile("auto")
                hw = profile["hardware"]
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                ollama_status = "🟢 Reachable" if ollama_is_running(base_url) else "🔴 Not reachable"
                return (
                    f"## Hardware Detection\n\n"
                    f"| Property | Value |\n|---|---|\n"
                    f"| CUDA | {'✅ Available — ' + str(hw.get('cuda_name', '')) if hw['cuda'] else '❌ Not found'} |\n"
                    f"| CUDA VRAM | {hw.get('cuda_vram_gb') or 'N/A'} GB |\n"
                    f"| Apple Metal | {'✅ Available' if hw['mps'] else '❌ Not found'} |\n"
                    f"| CPU threads | {hw['cpu_threads']} |\n"
                    f"| System RAM | {hw.get('system_memory_gb') or 'Unknown'} GB |\n"
                    f"| Ollama server | {ollama_status} |\n\n"
                    f"**Recommended profile:** `{profile['name'].upper()}`  "
                    f"— Context window: `{profile['context']} tokens`"
                )

            hw_btn.click(fn=_check_hw, inputs=[], outputs=hw_out)

    # ── Footer ─────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center;padding:20px 0 8px;color:#475569;font-size:0.8rem;">
        Resume_Crew v1.3.0 — grounded, local-first AI analysis.
        Keyword scores are not ATS simulations or hiring recommendations.
    </div>
    """)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PORT = int(os.getenv("GRADIO_PORT", "7860"))

    # Try ngrok first; fall back to Gradio's built-in sharing if no authtoken.
    ngrok_url = _setup_ngrok(PORT)
    use_share = (ngrok_url is None) and os.getenv("GRADIO_SHARE", "false").lower() == "true"

    # A public ngrok/share URL has no login by default — anyone with the link
    # can upload documents and run analysis. If both env vars are set, gate
    # the whole app behind basic auth so a shared link isn't wide open.
    auth_user = os.getenv("GRADIO_AUTH_USER", "").strip()
    auth_pass = os.getenv("GRADIO_AUTH_PASS", "").strip()
    auth = (auth_user, auth_pass) if auth_user and auth_pass else None
    if (ngrok_url or use_share) and not auth:
        print("   ⚠️  Sharing publicly with no login — set GRADIO_AUTH_USER/GRADIO_AUTH_PASS "
              "in .env to require a password.")

    print(f"\n🎯 Resume_Crew UI starting at http://localhost:{PORT}")
    if not ngrok_url:
        print("   (Set NGROK_AUTHTOKEN in .env for a live public URL)")

    demo.launch(
        server_port=PORT,
        share=use_share,
        show_error=True,
        inbrowser=True,
        auth=auth,
        **_LAUNCH_STYLE_KWARGS,
    )
