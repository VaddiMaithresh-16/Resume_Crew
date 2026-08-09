# Resume_Crew

Resume_Crew is a local-first application that compares a resume with a job description and creates an evidence-focused match report. It ships as both a **command-line tool** and a **Gradio web UI** with optional live ngrok sharing.

Supports PDF, DOCX, TXT, and Markdown inputs, deterministic keyword scoring, local Ollama models, and optional Gemini analysis.

## What it produces

Each analysis creates a new report folder containing:

- `match_report.md` — combined report
- `match_report.pdf` — PDF version of the combined report, the default download format (Gradio UI only, download button on Full Report tab)
- `match_report.docx` — Word version of the combined report, offered as a secondary download alongside the PDF
- `resume_profile.md` — factual candidate profile
- `job_description_profile.md` — job requirements profile
- `skills_gap_analysis.md` — strengths, evidence gaps, and interview risks
- `tailored_resume_bullets.md` — evidence-based bullet suggestions
- `interview_preparation.md` — role-specific questions and honest answer guidance
- `run_meta.json` — small sidecar (candidate, job title, score, timestamp) that powers the History tab

The deterministic score is a whole-term keyword overlap signal, not an ATS simulation or hiring recommendation.

## Requirements

- Python 3.10 or newer
- One LLM provider for full analysis:
  - **Ollama** — recommended for private, local processing
  - **Gemini** — optional cloud fallback; requires an API key and sends document text to Google

The hardware command works without Ollama or Gemini. Ranking now uses the LLM as well, so it needs a working provider like full analysis does.

## Project layout

```text
Resume_Crew/
├── src/resume_crew/         # Application package
├── tests/                   # Automated tests
├── samples/                 # Versioned demonstration documents
│   ├── Resumes/              #   sample_resume.pdf
│   └── Job_description/      #   sample_job_description.pdf
├── output/                  # Created locally for generated reports (ignored)
├── app.py                   # Gradio web UI entry point
├── main.py                  # CLI entry point (source-checkout)
├── .env.example             # Safe configuration template
└── pyproject.toml           # Package and dependency metadata
```

## Installation

Clone the repository first:

```bash
git clone https://github.com/<your-username>/Resume_Crew.git
cd Resume_Crew
```

### macOS

1. Install Python 3.10+ from [python.org](https://www.python.org/downloads/) or Homebrew.
2. In Terminal, from the project folder:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. For local analysis, install [Ollama](https://ollama.com/download), open it once, then run:

   ```bash
   ollama pull gemma3:4b
   ```

Apple Silicon is detected automatically. Check the recommended profile with `python main.py --check-hardware`.

### Windows (PowerShell)

1. Install Python 3.10+ from [python.org](https://www.python.org/downloads/) and select **Add Python to PATH**.
2. Open PowerShell in the project folder:

   ```powershell
   py -m venv venv
   .\venv\Scripts\Activate.ps1
   py -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once in that window, then activate again.

3. Install Ollama for Windows, then run:

   ```powershell
   ollama pull gemma3:4b
   ```

### Linux

On Debian/Ubuntu, install Python tooling if necessary:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
```

Then install the project:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install Ollama using its official instructions, then pull a model:

```bash
ollama pull gemma3:4b
```

For NVIDIA systems, confirm that the driver and `nvidia-smi` work before starting Ollama.

## Configuration

Copy the template:

```bash
cp .env.example .env
```

On Windows PowerShell, use:

```powershell
Copy-Item .env.example .env
```

Important settings:

```env
# Optional default input paths; command-line paths take precedence.
RESUME_PATH=./samples/Resumes/sample_resume.pdf
JD_PATH=./samples/Job_description/sample_job_description.pdf

# auto prefers local Ollama, then configured Gemini.
LLM_PROVIDER=auto
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b

# Required only for Gemini analysis.
GEMINI_API_KEY=
GEMINI_MODEL=gemini/gemini-3.1-flash-lite

# Raise on slow machines (WSL2, VMs) to avoid false 'Ollama not running' errors.
OLLAMA_TIMEOUT=3.0

# Gradio web UI server settings.
GRADIO_PORT=7860
GRADIO_SHARE=false

# Optional: paste your ngrok authtoken here for a live public URL when using the web UI.
# Get a free token at https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTHTOKEN=
```

Never commit `.env`; it is ignored by Git.

## Usage — Command Line (CLI)

Activate the virtual environment before each session, then run commands from the project root.

### Full local analysis

macOS/Linux:

```bash
python main.py \
  --resume samples/Resumes/sample_resume.pdf \
  --job-description samples/Job_description/sample_job_description.pdf \
  --provider ollama
```

Windows PowerShell:

```powershell
python main.py `
  --resume .\samples\Resumes\sample_resume.pdf `
  --job-description .\samples\Job_description\sample_job_description.pdf `
  --provider ollama
```

### Gemini analysis

After setting `GEMINI_API_KEY` in `.env`:

```bash
python main.py --resume ./my_resume.pdf --job-description ./job.docx --provider gemini
```

### Rank several resume versions with the LLM

```bash
python main.py --rank-resumes ./resume_versions --job-description ./job.docx --provider ollama
```

Each resume in the folder gets its own LLM scoring call (0-100 with a one-line note), so ranking
a large folder takes proportionally longer and, on `--provider gemini`, costs one API call per file.

### Hardware and model diagnostics

```bash
python main.py --check-hardware
python main.py --check-hardware --ollama-profile mps
python main.py --version
```

## Usage — Web UI (Gradio)

Start the browser interface from the project root:

```bash
python app.py
```

The UI opens automatically at `http://localhost:7860` and includes:

- **Analyze Resume tab** — upload resume + JD, pick provider, watch step-by-step progress, view results across 8 sub-tabs (Score, Resume Profile, Job Profile, Gap Analysis, Resume Bullets, Interview Prep, Resume Highlights, Full Report). Includes a **Cancel** button to stop a run between pipeline steps. The Full Report tab offers **PDF (default)** and **Word (.docx)** downloads of the combined report.
- **Rank Resumes tab** — score a local folder of resumes against a JD using the LLM, one lightweight scoring call per resume
- **Batch Analyze tab** — run the *full* 4-step analysis (not just a score) for every resume in a folder against one JD, saving a separate report per resume
- **Compare JDs tab** — score one resume against several job descriptions at once, to see which posting it fits best
- **History tab** — browse past saved runs, reload any report back into the tabs, and see a score-trend chart across runs
- **Hardware tab** — detect GPU, RAM, and Ollama status from the browser

### Live public URL via ngrok

Add your ngrok authtoken to `.env`:

```env
NGROK_AUTHTOKEN=your_token_here
```

Then run `python app.py`. A live public URL is printed in the terminal at startup:

```
============================================================
  🌐  Live ngrok URL: https://xxxx-xx-xx.ngrok-free.app
============================================================
```

Share this URL with anyone — no port-forwarding or VPN required.

Get a free authtoken at [dashboard.ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken).

**A public URL has no login by default** — anyone with the link can upload documents and run
analysis. Set both of these in `.env` to require a username/password before the app loads:

```env
GRADIO_AUTH_USER=someuser
GRADIO_AUTH_PASS=some-strong-password
```

### Port and share settings

```env
GRADIO_PORT=7860        # Change the local port
GRADIO_SHARE=true       # Use Gradio's built-in tunnel instead of ngrok
```

## Tracing and telemetry

CrewAI execution traces and telemetry are disabled by default in `.env.example` and in the application itself. The project passes `tracing=False` for every CrewAI task and disables CrewAI/OpenTelemetry telemetry. This keeps candidate data and execution metadata out of CrewAI tracing services.

## Privacy and data retention

- With `--provider ollama`, resume and job text stay on the local machine, subject to your local Ollama installation.
- With `--provider gemini`, source text is sent to Google; use it only when that is acceptable.
- Reports are created in `output/` and can contain personal information. The folder is ignored by Git.
- The current version does not persist analysis history.

## Troubleshooting

| Problem | Resolution |
|---|---|
| `ModuleNotFoundError` on startup | Activate the project virtual environment and run `pip install -r requirements.txt`. |
| `Ollama is not reachable` | Start the Ollama application/service, run `ollama pull gemma3:4b`, then retry. Raise `OLLAMA_TIMEOUT` in `.env` if on a slow machine. |
| Gemini fallback error | Add a valid `GEMINI_API_KEY` to `.env`, or use `--provider ollama`. |
| PDF has no extractable text | The PDF is likely scanned. Run OCR first, then supply the OCR result. |
| Unsupported file type | Convert the file to PDF, DOCX, TXT, or Markdown. |
| Input exceeds character limit | Split the document or remove irrelevant appendices. |
| PowerShell will not activate venv | Use the temporary execution-policy command in the Windows installation section. |
| Slow local analysis | Run `python main.py --check-hardware`; reduce Ollama context or use a smaller local model. |
| ngrok tunnel not created | Ensure `pyngrok` is installed (`pip install pyngrok`) and `NGROK_AUTHTOKEN` is set in `.env`. |
| Gradio port already in use | Set `GRADIO_PORT=7861` (or any free port) in `.env`. |
| `python app.py` fails with `DLL load failed... Application Control policy has blocked this file` (Windows) | A managed/corporate Windows security policy (WDAC/AppLocker) is blocking a native DLL — usually pandas, pulled in by Gradio — often because the project folder still has the "downloaded from the internet" flag. In PowerShell from the project root: `Get-ChildItem -Path .\venv -Recurse | Unblock-File`, then retry. If it still fails, move the project out of `Downloads` to a plain local folder (e.g. `C:\Resume_Crew`), delete `venv`, and reinstall. On a company-managed device the policy may be enforced centrally — ask IT to allow Python/pandas, or run the project inside WSL2 instead. |

## Development and verification

```bash
python -m pytest -q
python -m py_compile main.py app.py src/resume_crew/*.py
python -m pip check
```

The test suite covers document validation, keyword scoring (including single-char tokens like `R`), report structure, CRLF handling, output safety, timestamp format, and hardware-setting validation. A live LLM run requires an available Ollama server or valid Gemini credentials.
