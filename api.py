from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path

from dataclasses import replace
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from pydantic import BaseModel, Field

from script_to_doc.jobs import job_manager
from script_to_doc.pipeline import (
    PipelineConfig,
    ensure_nltk_resources,
    load_spacy_model,
    process_transcript,
    create_word_document,
)


BASE_DIR = Path(__file__).resolve().parent


def _ensure_writable_directory(path: Path, fallback_name: str) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError:
        fallback_root = Path(os.getenv("TMPDIR", "/tmp"))
        fallback_path = fallback_root / fallback_name
        fallback_path.mkdir(parents=True, exist_ok=True)
        return fallback_path


def _resolve_storage_dir(env_var: str, default_subdir: str) -> Path:
    explicit = os.getenv(env_var)
    if explicit:
        return _ensure_writable_directory(Path(explicit), Path(explicit).name)
    return _ensure_writable_directory(BASE_DIR / default_subdir, default_subdir)


UPLOAD_DIR = _resolve_storage_dir("UPLOAD_DIR", "uploads")
OUTPUT_DIR = _resolve_storage_dir("OUTPUT_DIR", "output_docs")

ensure_nltk_resources()
NLP_MODEL = load_spacy_model()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


CONFIG = PipelineConfig(
    use_llm_summary=_env_bool("USE_OPENAI_SUMMARY", False),
    use_llm_expander=_env_bool("USE_OPENAI_EXPANDER", False),
    use_llm_workflow=_env_bool("USE_OPENAI_WORKFLOW", False),
    llm_summary_model=os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4.1-mini"),
    llm_expander_model=os.getenv("OPENAI_EXPANDER_MODEL", "gpt-4.1-mini"),
    llm_workflow_model=os.getenv("OPENAI_WORKFLOW_MODEL", "gpt-4.1-mini"),
    llm_temperature=_env_float("OPENAI_TEMPERATURE", 0.2),
)

class PlanItem(BaseModel):
    title: str = Field("", description="Step title")
    summary: str = Field("", description="Short overview")
    details: str = Field("", description="Detailed instructions")
    actions: list[str] = Field(default_factory=list, description="Bullet point actions")


class PlanDocumentRequest(BaseModel):
    filename: str = Field("training_plan", description="Base filename without extension")
    title: Optional[str] = Field(None, description="Document title")
    tone: Optional[str] = Field(None, description="Preferred tone for the document")
    audience: Optional[str] = Field(None, description="Target audience for the document")
    plan: List[PlanItem] = Field(default_factory=list)


app = FastAPI(title="Transcript Trainer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <title>Transcript Trainer API</title>
        <style>
          body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 48px 24px; background: #0f172a; color: #e2e8f0; }
          main { max-width: 640px; margin: 0 auto; background: rgba(15, 23, 42, 0.75); border-radius: 24px; padding: 40px 36px; box-shadow: 0 24px 80px rgba(15, 23, 42, 0.45); }
          h1 { font-size: 2.25rem; margin-bottom: 0.5rem; }
          p { margin: 0.75rem 0; line-height: 1.6; }
          code, a { color: #38bdf8; }
        </style>
      </head>
      <body>
        <main>
          <h1>Transcript Trainer API</h1>
          <p>This endpoint powers the transcript-to-training document workflow.</p>
          <p>POST <code>/process</code> with a transcript file to start a job, then poll <code>/status/&lt;job_id&gt;</code> for progress or download the generated Word document from <code>/documents/&lt;filename&gt;</code>.</p>
          <p>For the full experience, open the web UI deployment or point your frontend at this API.</p>
        </main>
      </body>
    </html>
    """


def _store_upload(file: UploadFile, target_path: Path) -> Path:
    with target_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return target_path


def _slugify(name: str) -> str:
    cleaned = name.strip().lower()
    cleaned = cleaned.replace(".docx", "")
    cleaned = cleaned.replace(".txt", "")
    cleaned = cleaned.replace(" ", "-")
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch in {"-", "_"})
    return cleaned or f"document_{int(time.time())}"


def _run_job(job_id: str, transcript_path: Path) -> None:
    record = job_manager.get(job_id)
    tone = CONFIG.tone
    audience = CONFIG.audience
    if record and record.metadata:
        tone = record.metadata.get("tone", tone)
        audience = record.metadata.get("audience", audience)

    job_config = replace(CONFIG, tone=tone, audience=audience)

    job_manager.update(
        job_id,
        status="processing",
        current_step="load_transcript",
        progress=0.01,
    )

    try:
        result = process_transcript(
            transcript_path,
            OUTPUT_DIR,
            config=job_config,
            nlp=NLP_MODEL,
            progress_callback=lambda stage, pct: job_manager.update(
                job_id,
                status="processing",
                current_step=stage,
                progress=max(0.0, min(pct, 1.0)),
            ),
        )
    except Exception as exc:  # noqa: BLE001
        job_manager.fail(job_id, str(exc))
        return

    payload = {
        "transcript": str(result["transcript"]),
        "steps": result["steps"],
        "summary_sentences": result["summary_sentences"],
        "document_path": str(result["document_path"]),
        "plan": result.get("plan", []),
        "metrics": result.get("metrics", {}),
        "tone": tone,
        "audience": audience,
    }
    job_manager.complete(job_id, payload)


@app.post("/process")
async def process_endpoint(
    transcript: UploadFile = File(...),
    tone: str = Form("Neutral"),
    audience: str = Form("General"),
):
    if not transcript.filename or not transcript.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Please upload a .txt transcript file.")

    job = job_manager.create_job()

    timestamp = int(time.time())
    safe_stem = Path(transcript.filename).stem.replace(" ", "_")
    upload_path = UPLOAD_DIR / f"{safe_stem}_{timestamp}.txt"

    _store_upload(transcript, upload_path)
    job_manager.update(
        job.job_id,
        detail=str(upload_path.name),
        metadata={"tone": tone, "audience": audience},
    )

    thread = threading.Thread(target=_run_job, args=(job.job_id, upload_path), daemon=True)
    thread.start()

    return {"job_id": job.job_id}


@app.get("/documents/{filename}")
async def download_document(filename: str):
    target = OUTPUT_DIR / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="Document not found.")
    return FileResponse(target, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


@app.get("/status/{job_id}")
async def job_status(job_id: str, request: Request):
    record = job_manager.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found.")

    response = {
        "job_id": record.job_id,
        "status": record.status,
        "progress": record.progress,
        "current_step": record.current_step,
        "detail": record.detail,
        "error": record.error,
        "updated_at": record.updated_at,
    }

    if record.result:
        result = dict(record.result)
        document_path = result.get("document_path")
        if document_path:
            filename = Path(document_path).name
            result["doc_url"] = str(request.url_for("download_document", filename=filename))
        response["result"] = result

    response["metadata"] = record.metadata

    return response


@app.post("/generate-document")
async def generate_document(request: Request, payload: PlanDocumentRequest):
    if not payload.plan:
        raise HTTPException(status_code=400, detail="Plan data is required to generate a document.")

    sections: list[tuple[str, str]] = []
    for idx, item in enumerate(payload.plan, start=1):
        title = item.title.strip() or f"Step {idx}"
        summary = item.summary.strip()
        details = item.details.strip()
        actions = [action.strip() for action in item.actions if action.strip()]

        body_parts: list[str] = []
        if summary:
            body_parts.append(f"Overview: {summary}")
        if details:
            body_parts.append(details)
        if actions:
            bullet_lines = "\n".join(f"- {action}" for action in actions)
            body_parts.append(f"Key Actions:\n{bullet_lines}")

        sections.append((f"Step {idx}: {title}", "\n\n".join(body_parts)))

    filename = _slugify(payload.filename) + "_training"
    doc_path = create_word_document(
        filename,
        sections,
        OUTPUT_DIR,
        title=payload.title or CONFIG.document_title,
    )

    return {
        "document_path": str(doc_path),
        "doc_url": str(request.url_for("download_document", filename=doc_path.name)),
    }
