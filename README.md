# HireNex v6.0 — Reliability-first candidate screening and retrieval

HireNex is a local-first resume screening platform with a React/Vite frontend and FastAPI backend. It converts PDFs to Markdown with **PyMuPDF4LLM**, runs deterministic NLP and schema-validated LLM extraction, exports JSON/Excel, and supports recruiter search/RAG using ChromaDB.

## What changed in v6.0

- Combines the v4 clean-metrics and Excel-download fixes with the v5 Gemini/Groq provider layer.
- Preserves `APP######` identifiers embedded in source filenames. This is essential when evaluating shuffled labelled benchmark PDFs.
- Uses Gemini first for structured resume parsing and Groq as fallback by default; retrieval uses Groq first and Gemini as fallback. Routing is configurable.
- Retries transient API errors, waits on provider retry hints, paces calls, and falls back across configured providers/models.
- Checkpoints successful LLM records after each resume. A subsequent run reuses only unchanged successful records and retries failed/changed inputs.
- Writes `output/llm/llm_failures.json` with application ID, source Markdown filename, and final error; the UI can download the full report.
- Displays the first failure details in the processing UI instead of only a failure count.
- Invalidates old evaluation output when data or parser outputs change.
- Uses PyMuPDF4LLM for PDF-to-Markdown conversion with OCR enabled as a fallback (`use_ocr=True`, `force_ocr=False`).

> No software can honestly guarantee 100% extraction accuracy for arbitrary resumes. The reliability goal is to attempt every input, preserve successful work, retry recoverable failures, and clearly report files that cannot be processed. Human review remains necessary for recruitment decisions.

## Requirements

- Windows 10/11, macOS, or Linux
- Python 3.11+ recommended
- Node.js LTS and npm
- Gemini API key, Groq API key, or both
- For scanned/image-only PDFs, install Tesseract OCR and make sure the executable is on `PATH` if your PyMuPDF4LLM installation requires it.

## 1. Backend setup (first time only)

Open a terminal in the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Add at least one real API key in `.env`:

```env
GEMINI_API_KEY=your_real_gemini_key
GROQ_API_KEY=your_real_groq_key
```

You may use one provider only, but two configured providers give the parser a fallback. Never put API keys in frontend files or commit `.env`.

## 2. Frontend setup (first time only)

Open a second terminal:

```powershell
cd frontend
npm install
```

## 3. Run HireNex (every subsequent time)

**Terminal 1 — backend** (from the project root):

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --reload
```

**Terminal 2 — frontend**:

```powershell
cd frontend
npm run dev
```

Open:

- Website: http://localhost:5173
- API health/docs: http://127.0.0.1:8000/docs

Keep both terminals open while using the application. Stop each server with `Ctrl+C`.

On Windows, after the first-time setup you can also launch both services from the project root with:

```powershell
.\\start_hirenex.ps1
```

If PowerShell already shows `(.venv)`, do not activate it again. If the activation script is missing, recreate the environment using the first-time setup commands.

## 4. Recommended workflow

1. Upload individual PDF resumes or a ZIP containing PDFs.
2. Wait for PDF → Markdown conversion to finish. Check failed conversions before proceeding.
3. Run NLP parsing.
4. Run LLM parsing. The parser runs sequentially and prioritizes completeness over throughput. Successful records are checkpointed; rerunning retries only failed or changed Markdown inputs.
5. Download JSON and Excel outputs.
6. Build/search the retrieval index and ask recruiter questions.
7. Run Evaluation after the dataset and parser outputs are final.

For a benchmark archive whose filenames contain `APP######`, start with an empty HireNex dataset (use **Clear Data** only if you are willing to remove the current local dataset), then upload the archive. This preserves the benchmark IDs and avoids false metric comparisons.

## 5. Reliability and troubleshooting

### LLM key missing
- Confirm `.env` is in the project root beside `llm.py` and `ai_providers.py`.
- Add at least one real `GEMINI_API_KEY` or `GROQ_API_KEY`.
- Restart the backend after editing `.env`.
- The `/api/providers` endpoint reports which providers are configured without exposing keys.

### LLM parsing has failures
- Inspect `output/llm/llm_failures.json`.
- Check backend terminal logs for API authentication, quota/429, timeout, unavailable model, and malformed JSON errors.
- If your API plan is more restrictive, increase `GEMINI_MIN_INTERVAL_SECONDS` / `GROQ_MIN_INTERVAL_SECONDS`.
- If a provider is unavailable, configure both providers so fallback is possible.
- Run **Process LLM** again: unchanged successful records are reused and failed records are retried.

### OCR / conversion
- Digital PDFs are text-extracted; OCR is enabled as a fallback for scanned pages.
- If image-only PDFs fail, verify Tesseract is installed and available on `PATH`.
- Check `conversion` status and the backend logs for the exact file-level conversion error.

### Frontend says Vite is not recognized
Run `npm install` once inside `frontend`, then `npm run dev`.

### Evaluation scores are unexpectedly low
- Ensure the benchmark filenames contain the expected `APP######` IDs.
- Clear the old dataset before uploading a new labelled benchmark if there are ID conflicts.
- Run conversion, NLP/LLM parsing, then evaluation again.
- Do not interpret record coverage as extraction accuracy. Review field-level precision, recall, F1, and numeric error metrics.

## 6. Data and privacy

Uploaded resumes, manifest data, generated Markdown, parser outputs, ChromaDB files, and `.env` are local runtime data and are excluded from Git by `.gitignore`. Do not publish real candidate resumes, contact details, API keys, or private evaluation data in a public repository. Use synthetic/anonymized resumes for demos.

## Project structure

```text
HireNex/
├── backend/
│   └── main.py
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── evaluation/
│   ├── ground_truth.json
│   └── recruiter_queries.json
├── sample_data/
│   └── HireNex_90_Resumes_Ordered.zip
├── ai_providers.py
├── experience_utils.py
├── llm.py
├── nlp.py
├── retrival.py
├── requirements.txt
├── .env.example
└── VERSION.txt
```

## Model defaults

The default model IDs are configurable in `.env`:
- Gemini parsing: `gemini-3.8-flash`, fallback `gemini-3.5-flash-lite`
- Groq parsing/search/answer: `openai/gpt-oss-120b`, fallback `openai/gpt-oss-20b`

Provider access and quotas depend on the API account. Verify your configured models with the **System → Provider test** panel before processing a large batch.
