"""
HireNex - LLM Resume Parser

Responsibilities of this module:
    - Read ONLY Markdown resumes from md_data/
    - Extract structured resume data with an LLM (Gemini or Groq, chosen by
      ai_providers.py - Gemini first for parsing, Groq as automatic fallback)
    - Save results as JSON and Excel

All provider, model, retry and fallback logic lives in ai_providers.py.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, List

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from ai_providers import LLMMeta, LLMUnavailableError, configured_providers, generate_json, provider_order


# -----------------------------------------------------------------------------
# PYDANTIC SCHEMA
# -----------------------------------------------------------------------------

class _Lenient(BaseModel):
    """Tolerate sloppy-but-recoverable model output.

    Gemini returns schema-constrained JSON, but Groq's JSON mode only guarantees
    syntactically valid JSON. So: nulls fall back to defaults, numbers given
    where text is expected become text, and list items are coerced to strings.
    """

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        cleaned: dict[str, Any] = {}
        for key, value in data.items():
            if value is None:
                continue
            info = cls.model_fields.get(key)
            annotation = info.annotation if info else None
            if annotation is str and isinstance(value, (int, float, bool)):
                value = str(value)
            elif annotation == List[str] and isinstance(value, list):
                value = [str(item) for item in value if isinstance(item, (str, int, float))]
            elif annotation == List[str] and isinstance(value, str):
                value = [value] if value.strip() else []
            cleaned[key] = value
        return cleaned

class Contact(_Lenient):
    email: str = Field(default="")
    phone: str = Field(default="")
    linkedin: str = Field(default="")
    github: str = Field(default="")
    portfolio: str = Field(default="")


class Score(_Lenient):
    value: str = Field(default="")
    type: str = Field(default="")


class Education(_Lenient):
    degree: str = Field(default="")
    field: str = Field(default="")
    score: Score = Field(default_factory=Score)
    institute: str = Field(default="")
    start_year: str = Field(default="")
    end_year: str = Field(default="")


class Project(_Lenient):
    name: str = Field(default="")
    description: str = Field(default="")
    technologies: List[str] = Field(default_factory=list)
    github: str = Field(default="")
    year: str = Field(default="")


class Experience(_Lenient):
    title: str = Field(
        default="",
        description="Professional role/job title held"
    )
    Company: str = Field(
        default="",
        description="Name of the employer or organization"
    )
    duration: str = Field(
        default="",
        description="Employment period exactly as supported by the resume"
    )
    responsibilities: List[str] = Field(
        default_factory=list,
        description="Key responsibilities supported by the resume"
    )

    @model_validator(mode="before")
    @classmethod
    def _company_alias(cls, data: Any) -> Any:
        if isinstance(data, dict) and "Company" not in data:
            for alias in ("company", "employer", "organization"):
                if data.get(alias):
                    data = {**data, "Company": data[alias]}
                    break
        return data


class Certification(_Lenient):
    name: str = Field(default="")
    issuer: str = Field(default="")
    year: str = Field(default="")
    credential_url: str = Field(default="")


class ResumeData(_Lenient):
    name: str = Field(
        default="",
        description="Full candidate name"
    )
    email: str = Field(
        default="",
        description="Primary email address"
    )
    phone: str = Field(default="")
    linkedin: str = Field(default="")
    github: str = Field(default="")
    portfolio: str = Field(default="")
    tag: str = Field(default="LLM")
    personal_summary: str = Field(default="")
    skills: List[str] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def application_no_from_path(path: Path) -> str:
    return path.stem.upper()


def clean_result(record: ResumeData) -> dict[str, Any]:
    """Convert Pydantic output into the common HireNex JSON structure."""
    data = record.model_dump()

    experience = []
    for item in data.get("experience", []):
        experience.append({
            "company": item.get("Company", ""),
            "role": item.get("title", ""),
            "description": " ".join(item.get("responsibilities", [])),
            "start_year": "",
            "end_year": "",
            "duration": item.get("duration", ""),
        })

    return {
        "candidate": {
            "name": data.get("name", ""),
            "tag": data.get("tag", "LLM"),
            "contact": {
                "email": data.get("email", ""),
                "phone": data.get("phone", ""),
                "linkedin": data.get("linkedin", ""),
                "github": data.get("github", ""),
                "portfolio": data.get("portfolio", ""),
            },
            "personal_summary": data.get("personal_summary", ""),
        },
        "education": [item for item in data.get("education", [])],
        "skills": data.get("skills", []),
        "projects": [item for item in data.get("projects", [])],
        "experience": experience,
        "certifications": [item for item in data.get("certifications", [])],
        "achievements": data.get("achievements", []),
    }


SYSTEM_PROMPT = """
You are the HireNex resume extraction engine.

Parse the supplied Markdown resume into ONE JSON object.

Rules:
1. Extract only information supported by the resume.
2. Never invent a value.
3. When a scalar value is missing, return an empty string.
4. When a list has no data, return an empty list.
5. Preserve names, organizations, skills, project titles, dates and links faithfully.
6. Extract ALL clearly supported skills, education entries, projects, experience entries,
   certifications and achievements rather than returning only a few examples.
7. For experience, preserve the original duration text where present.
8. Do not infer years or responsibilities that the resume does not support.
9. Keep project descriptions complete and do not split one project into multiple projects merely
   because it contains several bullet points.
10. Return only data that belongs to this resume.
11. Every value must be a string (or a list / object as shown). Never use null.

Return ONLY a JSON object with exactly this shape (no commentary, no markdown fences):
""" + json.dumps({
    "name": "", "email": "", "phone": "", "linkedin": "", "github": "", "portfolio": "",
    "tag": "LLM", "personal_summary": "",
    "skills": [""],
    "education": [{"degree": "", "field": "", "score": {"value": "", "type": ""},
                   "institute": "", "start_year": "", "end_year": ""}],
    "projects": [{"name": "", "description": "", "technologies": [""], "github": "", "year": ""}],
    "experience": [{"title": "", "Company": "", "duration": "", "responsibilities": [""]}],
    "certifications": [{"name": "", "issuer": "", "year": "", "credential_url": ""}],
    "achievements": [""],
}, indent=1)


def _validate_resume(text: str) -> ResumeData:
    from ai_providers import extract_json

    return ResumeData.model_validate(extract_json(text))


def extract_resume_data(resume_text: str) -> tuple[dict[str, Any], LLMMeta]:
    """
    Extract structured data from one Markdown resume.

    Gemini is tried first (schema-constrained output); Groq is the automatic
    fallback. Returns the HireNex record plus metadata describing which
    provider/model actually produced it.

    The application number is deliberately NOT part of the schema, so the LLM
    can never alter it; process_resumes() assigns it.
    """
    parsed, meta = generate_json(
        "parsing",
        SYSTEM_PROMPT,
        "RESUME (Markdown):\n\n" + resume_text,
        schema=ResumeData,
        validator=_validate_resume,
        temperature=0.1,
        reasoning="low",
    )
    return clean_result(parsed), meta


# -----------------------------------------------------------------------------
# EXCEL OUTPUT
# -----------------------------------------------------------------------------


def flatten_for_excel(record: dict[str, Any]) -> dict[str, Any]:
    candidate = record.get("candidate", {})
    contact = candidate.get("contact", {})

    def as_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value or "")

    return {
        "Application_No": record.get("application_no", ""),
        "Name": candidate.get("name", ""),
        "Tag": candidate.get("tag", ""),
        "Email": contact.get("email", ""),
        "Phone": contact.get("phone", ""),
        "LinkedIn": contact.get("linkedin", ""),
        "GitHub": contact.get("github", ""),
        "Portfolio": contact.get("portfolio", ""),
        "Personal_Summary": candidate.get("personal_summary", ""),
        "Education": as_json(record.get("education", [])),
        "Skills": ", ".join(record.get("skills", [])),
        "Projects": as_json(record.get("projects", [])),
        "Experience": as_json(record.get("experience", [])),
        "Certifications": as_json(record.get("certifications", [])),
        "Achievements": as_json(record.get("achievements", [])),
    }


def save_outputs(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "resumes.json"
    excel_path = output_dir / "resumes.xlsx"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame([flatten_for_excel(record) for record in records])
    df.to_excel(excel_path, index=False)

    return {
        "records": len(records),
        "json_path": str(json_path),
        "excel_path": str(excel_path),
    }


# -----------------------------------------------------------------------------
# BATCH PROCESSOR USED BY app.py
# -----------------------------------------------------------------------------


def process_resumes(
    input_dir: str = "md_data",
    output_dir: str = "output/llm",
    progress_callback: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Parse Markdown resumes with checkpointing and retry of transient failures.

    Successful records are atomically checkpointed after each resume. A later
    run skips only records whose Markdown content hash is unchanged; failed or
    changed resumes are attempted again. The final JSON remains a plain list so
    existing retrieval/evaluation code stays compatible.
    """
    md_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not md_dir.exists():
        raise FileNotFoundError(f"Markdown directory not found: {md_dir}")

    md_files = sorted(md_dir.glob("APP*.md"))
    if not md_files:
        raise FileNotFoundError(f"No APP*.md files found in {md_dir}")

    if not configured_providers():
        raise LLMUnavailableError(
            "No LLM API key is configured. Add GEMINI_API_KEY and/or GROQ_API_KEY to your .env file."
        )

    json_path = out_dir / "resumes.json"
    failures_path = out_dir / "llm_failures.json"
    existing: dict[str, dict[str, Any]] = {}
    if json_path.exists():
        try:
            old_records = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(old_records, list):
                existing = {
                    str(record.get("application_no")): record
                    for record in old_records
                    if isinstance(record, dict) and record.get("application_no")
                }
        except (OSError, json.JSONDecodeError):
            existing = {}

    # Keep only unchanged successful records belonging to the current input set.
    current_hashes = {
        md_path.name: hashlib.sha256(md_path.read_bytes()).hexdigest()
        for md_path in md_files
    }
    records: dict[str, dict[str, Any]] = {}
    for md_path in md_files:
        app_no = application_no_from_path(md_path)
        old = existing.get(app_no)
        metadata = old.get("metadata", {}) if isinstance(old, dict) else {}
        if (
            old
            and metadata.get("processing_status") == "success"
            and metadata.get("source_markdown") == md_path.name
            and metadata.get("source_sha256") == current_hashes[md_path.name]
        ):
            records[app_no] = old

    errors: list[dict[str, str]] = []
    usage: dict[str, int] = {}
    total = len(md_files)
    route = " → ".join(p.title() for p in provider_order("parsing"))
    completed = len(records)

    def report(done: int, current: str, message: str, phase: str = "processing") -> None:
        if progress_callback:
            progress_callback(
                total=total,
                completed=done,
                failed=len(errors),
                current=current,
                message=message,
                phase=phase,
            )

    def checkpoint() -> None:
        ordered = [records[key] for key in sorted(records)]
        temporary = json_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(ordered, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(json_path)

    report(completed, "", f"Preparing reliable LLM parser ({route})…", "initializing")

    for md_path in md_files:
        app_no = application_no_from_path(md_path)
        if app_no in records:
            report(completed, md_path.name, f"Already parsed {app_no}; checkpoint reused")
            continue

        report(completed, md_path.name, f"Parsing {app_no} with {route}…")
        resume_text = md_path.read_text(encoding="utf-8", errors="replace")
        source_hash = current_hashes[md_path.name]
        last_error = ""

        # Provider layer already retries and falls back. This outer recovery pass
        # handles transient exhaustion after the provider cooldown is cleared.
        for attempt in range(3):
            try:
                parsed, meta = extract_resume_data(resume_text)
                parsed["application_no"] = app_no
                parsed["metadata"] = {
                    "source_markdown": md_path.name,
                    "source_sha256": source_hash,
                    "processing_status": "success",
                    "parser": "LLM",
                    "provider": meta.provider,
                    "model": meta.model,
                    "fallback_used": meta.fallback_used,
                    "provider_errors": meta.errors,
                    "attempts": attempt + 1,
                }
                records[app_no] = parsed
                label = f"{meta.provider}:{meta.model}"
                usage[label] = usage.get(label, 0) + 1
                checkpoint()
                last_error = ""
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                lowered = last_error.lower()
                transient = any(token in lowered for token in (
                    "429", "rate limit", "resource_exhausted", "503", "502",
                    "504", "timeout", "timed out", "connection", "temporarily",
                    "cooldown", "cool down",
                ))
                if not transient or attempt >= 2:
                    break
                try:
                    from ai_providers import reset_provider_state
                    reset_provider_state()
                except Exception:
                    pass
                wait_seconds = (15, 45)[attempt]
                report(completed, md_path.name, f"Temporary provider failure; retrying {app_no} in {wait_seconds}s", "retrying")
                time.sleep(wait_seconds)

        if last_error:
            errors.append({"application_no": app_no, "source_markdown": md_path.name, "error": last_error})
            failures_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
        completed += 1
        report(completed, md_path.name, f"LLM attempted {completed} of {total}")

    ordered_records = [records[key] for key in sorted(records)]
    output = save_outputs(ordered_records, out_dir)
    failures_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
    output["errors"] = errors
    output["failed"] = len(errors)
    output["skipped_existing"] = sum(
        1 for md_path in md_files
        if application_no_from_path(md_path) in existing
        and application_no_from_path(md_path) in records
        and existing[application_no_from_path(md_path)].get("metadata", {}).get("source_sha256")
            == current_hashes[md_path.name]
    )
    usage = {}
    for record in ordered_records:
        metadata = record.get("metadata", {}) if isinstance(record, dict) else {}
        provider, model = metadata.get("provider"), metadata.get("model")
        if provider and model:
            label = f"{provider}:{model}"
            usage[label] = usage.get(label, 0) + 1
    output["models_used"] = usage
    output["failures_path"] = str(failures_path)
    output["coverage"] = {
        "total": total,
        "successful": len(ordered_records),
        "failed": len(errors),
        "percentage": round((len(ordered_records) / total) * 100, 2) if total else 0,
    }
    report(total, "", "Saving final LLM JSON and Excel outputs…", "saving")
    return output


if __name__ == "__main__":
    result = process_resumes()
    print(json.dumps(result, indent=2, ensure_ascii=False))
