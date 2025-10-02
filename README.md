# ScriptToDoc

Transform meeting transcripts into structured training material and Word documents. The project contains a Python NLP pipeline plus a Next.js/MUI frontend for a polished user experience.

## Prerequisites

- Python 3.9+
- Node.js 18+

## Python pipeline

1. **Create the virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the CLI**
   ```bash
   .venv/bin/python main.py <input_dir> <output_dir>
   ```
   Add `--json` to emit machine-readable output. Generated `.docx` files land in `<output_dir>`.

Key modules:

- `script_to_doc.pipeline` supplies modular helpers:
  - `load_transcripts`, `clean_transcript` (timestamp + filler removal, sentence tokenization)
  - `summarize_transcript` (TextRank summariser)
  - `extract_steps` (spaCy imperative detection)
  - `expand_steps`, `create_word_document`
  - `process_transcript`, `run_pipeline` for orchestration
- `PipelineConfig` centralises tunable settings (summary ratio, expander hook, document title).
- `api.py` exposes the FastAPI backend with `/process` (file upload) and `/documents/{filename}` for downloads.

The Word export uses `python-docx`, which lets us instantiate `Document()` and add headings/paragraphs programmatically [python-docx Quickstart](https://python-docx.readthedocs.io/en/latest/user/quickstart.html). Summaries rely on a TextRank implementation that extracts the most important sentences from the transcript [TextRank overview](https://radimrehurek.com/gensim_3.8.3/summarization/summariser.html).

### Run the web API

```bash
.venv/bin/uvicorn api:app --reload
```

Set `PORT`/`HOST` with `--port`/`--host` flags as needed. The server writes uploads to `uploads/` and generated docs to `output_docs/`.

Once a plan is loaded you can POST it back to `/generate-document` to rebuild a `.docx` without reprocessing the transcript. Provide a JSON payload of the form (tone/audience are optional overrides):

```json
{
  "filename": "custom_training",
  "title": "Training Document from Meeting Transcript",
  "tone": "Neutral",
  "audience": "General",
  "plan": [
    {
      "title": "Step 1",
      "summary": "Short overview",
      "details": "Detailed instructions",
      "actions": ["Bullet action"]
    }
  ]
}
```

#### Enable OpenAI models (optional)

Set the following environment variables before launching Uvicorn to tap into OpenAI for abstractive summarisation and richer training copy:

```bash
export OPENAI_API_KEY=sk-...
export USE_OPENAI_SUMMARY=true          # turn on LLM summary extraction
export USE_OPENAI_EXPANDER=true         # turn on LLM step expansion
export USE_OPENAI_WORKFLOW=true         # let OpenAI generate the full training plan
export OPENAI_SUMMARY_MODEL=gpt-4.1-mini
export OPENAI_EXPANDER_MODEL=gpt-4.1-mini
export OPENAI_WORKFLOW_MODEL=gpt-4.1-mini
export OPENAI_TEMPERATURE=0.2
.venv/bin/uvicorn api:app --reload
```

If any OpenAI call fails (missing key, quota, etc.) the pipeline automatically falls back to the deterministic TextRank + template approach.

## Frontend (Next.js + Tailwind)

```bash
cd transcript-trainer-ui
npm install
npm run dev
```

Expose the Python API URL via `NEXT_PUBLIC_API_BASE_URL` (defaults to `http://localhost:8000`). The UI provides:

- Drag-and-drop transcript upload with inline validation (5 MB cap)
- Real-time progress feedback for upload, AI summarisation, expansion, and document generation
- Tone and audience controls that feed the OpenAI prompts
- Editable training plan cards (titles, overviews, details, bullet actions) with undo history and reset-to-original for each step
- Live document preview highlighting differences from the original LLM draft before export
- Download button that regenerates the Word document from your edited plan on demand
- Run metrics (duration + token usage) surfaced once processing completes

The `/process` endpoint now returns a job identifier immediately. The frontend polls `/status/{job_id}` to keep the linear progress indicator and stepper text synced with backend progress.

## Development tips

- Adjust filler words via `PipelineConfig.filler_words` or swap in a custom `step_expander` callable to integrate LLM-generated explanations.
- Swap summarisation libraries by adapting `text_rank_summarize` within `pipeline.py`, or rely on `USE_OPENAI_WORKFLOW` to have OpenAI return a structured set of steps directly.
- For production, restrict CORS origins in `api.py`, rotate job IDs regularly, and secure file storage.

## Testing

- Backend: `pytest`
- Frontend integration (document export payload):

```bash
cd transcript-trainer-ui
npm run test
```

## Sample data

A sample file lives under `sample_data/transcripts/`. Run the CLI with:

```bash
.venv/bin/python main.py sample_data/transcripts output_docs
```

Verify the generated Word file inside `output_docs/` and inspect CLI output for processed steps.
