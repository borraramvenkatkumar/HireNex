from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

from experience_utils import calculate_total_experience

BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "evaluation"
GROUND_TRUTH_PATH = EVAL_DIR / "ground_truth.json"
QUERIES_PATH = EVAL_DIR / "recruiter_queries.json"
RESULT_PATH = EVAL_DIR / "latest_evaluation.json"

NLP_JSON = BASE_DIR / "output" / "nlp" / "resumes.json"
LLM_JSON = BASE_DIR / "output" / "llm" / "resumes.json"
MANIFEST = BASE_DIR / "data" / "resume_manifest.json"
MD_DIR = BASE_DIR / "md_data"


def load_json(path: Path, default):
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9+#./@ -]", "", text)
    return text.strip()


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def flatten_strings(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        if value.strip():
            out.append(value.strip())
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(flatten_strings(v))
    elif isinstance(value, list):
        for v in value:
            out.extend(flatten_strings(v))
    return out


def dedupe_norm(items: list[Any]) -> set[str]:
    result = set()
    for item in items:
        value = normalize_text(item)
        if value:
            result.add(value)
    return result


def get_application_no(record: dict[str, Any]) -> str:
    return str(record.get("application_no") or record.get("Application_No") or "").upper()


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def extract_years_from_text(text: str) -> float | None:
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", text.lower())
    if match:
        return float(match.group(1))
    return None


def extract_predicted_experience(record: dict[str, Any]) -> float | None:
    experience = record.get("experience", [])
    candidate = record.get("candidate", {}) or {}
    # Prefer explicitly structured numeric duration fields when available.
    duration_keys = {"years", "years_of_experience", "total_years", "total_experience_years", "experience_years"}
    def structured_years(value: Any) -> float | None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower().strip() in duration_keys:
                    parsed = parse_number(item)
                    if parsed is not None:
                        return parsed
            for item in value.values():
                found = structured_years(item)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = structured_years(item)
                if found is not None:
                    return found
        return None
    structured = structured_years(experience)
    if structured is not None:
        return structured
    # Same calculation retrieval uses: merge date ranges, no double counting.
    entries = [item for item in as_list(experience) if isinstance(item, dict)]
    if entries:
        total = calculate_total_experience(entries)
        if total > 0:
            return total
    values: list[str] = []
    values.extend(flatten_strings(experience))
    values.extend(flatten_strings(candidate.get("personal_summary", "")))
    values.extend(flatten_strings(record.get("summary", "")))
    for value in values:
        years = extract_years_from_text(value)
        if years is not None:
            return years
    return None


DEGREE_RE = re.compile(r"b\.?\s?tech|b\.?\s?e\b|bachelor|m\.?\s?tech|m\.?\s?e\b|master|b\.?\s?sc|m\.?\s?sc|degree", re.I)


def extract_predicted_cgpa(record: dict[str, Any]) -> float | None:
    """CGPA of the candidate's degree.

    Both parsers store grades as education[i].score = {"value": "8.5", "type": "CGPA"}.
    10th / 12th entries often carry grades too, so degree-level entries win.
    """
    found: list[tuple[bool, float]] = []
    for education in as_list(record.get("education", [])):
        if isinstance(education, str):
            education = {"degree": education}
        if not isinstance(education, dict):
            continue
        value: float | None = None
        score = education.get("score")
        if isinstance(score, dict) and re.search(r"cgpa|gpa", str(score.get("type", "")), re.I):
            value = parse_number(score.get("value"))
        if value is None:
            for key, item in education.items():
                if str(key).lower().strip() in {"cgpa", "gpa", "grade_point_average"}:
                    value = parse_number(item)
                    break
        if value is None:
            match = re.search(r"(?:cgpa|gpa)\s*[:=-]?\s*(\d+(?:\.\d+)?)", " ".join(flatten_strings(education)), re.I)
            value = float(match.group(1)) if match else None
        if value is not None:
            is_degree = bool(DEGREE_RE.search(" ".join(str(education.get(k, "")) for k in ("degree", "field"))))
            found.append((is_degree, value))
    if not found:
        return None
    degree_entries = [v for is_degree, v in found if is_degree]
    return (degree_entries or [v for _, v in found])[0]


def extract_predicted_college(record: dict[str, Any]) -> str:
    for education in as_list(record.get("education", [])):
        if isinstance(education, dict):
            for key in ("institution", "college", "university", "school", "College", "Institution"):
                if education.get(key):
                    return str(education[key])
    return ""


def extract_predicted_degree(record: dict[str, Any]) -> str:
    for education in as_list(record.get("education", [])):
        if isinstance(education, dict):
            for key in ("degree", "Degree", "qualification", "course", "program"):
                if education.get(key):
                    return str(education[key])
    return ""


def extract_predicted_record(record: dict[str, Any]) -> dict[str, Any]:
    candidate = record.get("candidate", {}) or {}
    contact = candidate.get("contact", {}) or {}
    education_text = " ".join(flatten_strings(record.get("education", [])))
    projects = record.get("projects", [])
    project_texts = flatten_strings(projects)
    certifications = flatten_strings(record.get("certifications", []))
    return {
        "application_no": get_application_no(record),
        "name": candidate.get("name") or record.get("name", ""),
        "email": contact.get("email") or record.get("email", ""),
        "phone": contact.get("phone") or record.get("phone", ""),
        "college": extract_predicted_college(record) or education_text,
        "degree": extract_predicted_degree(record) or education_text,
        "skills": as_list(record.get("skills", [])),
        "certifications": certifications,
        "projects": project_texts,
        "experience_years": extract_predicted_experience(record),
        "cgpa": extract_predicted_cgpa(record),
    }


def scalar_match(pred: Any, truth: Any) -> bool:
    p = normalize_text(pred)
    t = normalize_text(truth)
    return bool(p) and bool(t) and p == t


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def numeric_report(errors: list[float], expected: int, within: int) -> dict[str, Any]:
    """MAE / RMSE over pairs where both values exist; misses are reported, not imputed."""
    mae = sum(abs(e) for e in errors) / len(errors) if errors else None
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else None
    return {
        "mae": round(mae, 4) if mae is not None else None,
        "rmse": round(rmse, 4) if rmse is not None else None,
        "accuracy": round(within / expected, 4) if expected else 0.0,
        "compared": len(errors),
        "expected": expected,
        "missing": expected - len(errors),
    }


def model_usage(records: list[Any]) -> dict[str, int]:
    """Which LLM provider/model produced the parsed records (LLM parser only)."""
    usage: dict[str, int] = {}
    for record in records:
        meta = record.get("metadata") if isinstance(record, dict) else None
        if isinstance(meta, dict) and meta.get("provider"):
            label = f"{meta['provider']}:{meta.get('model', '')}"
            usage[label] = usage.get(label, 0) + 1
    return usage


def evaluate_parser(prediction_path: Path, parser_name: str, ground_truth: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare one parser's output with the labeled benchmark.

    Skills / certifications: micro-averaged precision, recall and F1.
    Name / email / phone:    exact normalized match, reported per field.
    Experience / CGPA:       MAE and RMSE on pairs where both values exist, plus
                             accuracy within a tolerance (0.25 years / 0.1 CGPA).
    """
    predicted_raw = load_json(prediction_path, [])
    if isinstance(predicted_raw, dict):
        for key in ("resumes", "records", "candidates"):
            if isinstance(predicted_raw.get(key), list):
                predicted_raw = predicted_raw[key]
                break
    predicted_raw = [r for r in predicted_raw if isinstance(r, dict)]
    pred_by_app = {get_application_no(r): extract_predicted_record(r) for r in predicted_raw}
    truth_by_app = {str(r.get("application_no", "")).upper(): r for r in ground_truth}

    total = len(truth_by_app)
    coverage = len(set(pred_by_app) & set(truth_by_app)) / total if total else 0.0

    contact_hits = {"name": 0, "email": 0, "phone": 0}
    skill_tp = skill_fp = skill_fn = 0
    cert_tp = cert_fp = cert_fn = 0
    exp_errors: list[float] = []
    cgpa_errors: list[float] = []
    exp_expected = cgpa_expected = exp_within = cgpa_within = 0

    for app_no, truth in truth_by_app.items():
        pred = pred_by_app.get(app_no, {})

        for field in contact_hits:
            if scalar_match(pred.get(field), truth.get(field)):
                contact_hits[field] += 1

        ps = dedupe_norm(as_list(pred.get("skills", [])))
        ts = dedupe_norm(as_list(truth.get("skills", [])))
        skill_tp, skill_fp, skill_fn = skill_tp + len(ps & ts), skill_fp + len(ps - ts), skill_fn + len(ts - ps)

        pc = dedupe_norm(as_list(pred.get("certifications", [])))
        tc = dedupe_norm(as_list(truth.get("certifications", [])))
        cert_tp, cert_fp, cert_fn = cert_tp + len(pc & tc), cert_fp + len(pc - tc), cert_fn + len(tc - pc)

        texp = parse_number(truth.get("experience_years"))
        if texp is not None:
            exp_expected += 1
            if pred.get("experience_years") is not None:
                error = float(pred["experience_years"]) - texp
                exp_errors.append(error)
                exp_within += abs(error) <= 0.25

        tcgpa = parse_number(truth.get("cgpa"))
        if tcgpa is not None:
            cgpa_expected += 1
            if pred.get("cgpa") is not None:
                error = float(pred["cgpa"]) - tcgpa
                cgpa_errors.append(error)
                cgpa_within += abs(error) <= 0.1

    contact = {field: round(hits / total, 4) if total else 0.0 for field, hits in contact_hits.items()}
    contact["average"] = round(sum(contact.values()) / 3, 4)
    skills, certs = prf(skill_tp, skill_fp, skill_fn), prf(cert_tp, cert_fp, cert_fn)
    experience = numeric_report(exp_errors, exp_expected, exp_within)
    cgpa = numeric_report(cgpa_errors, cgpa_expected, cgpa_within)

    # One headline number for side-by-side comparison: the plain mean of the five
    # components. It is a custom summary, not a standard F1-score.
    overall = (skills["f1"] + certs["f1"] + contact["average"] + experience["accuracy"] + cgpa["accuracy"]) / 5

    return {
        "parser": parser_name,
        "records_expected": total,
        "records_predicted": len(pred_by_app),
        "coverage": round(coverage, 4),
        "contact": contact,
        "skills": skills,
        "certifications": certs,
        "experience": experience,
        "cgpa": cgpa,
        "overall_score": round(overall, 4),
        "models_used": model_usage(predicted_raw),
    }


def binary_ndcg(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, app_no in enumerate(retrieved[:k], start=1) if app_no in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def retrieval_metrics(predicted: list[str], expected: list[str], k: int = 10) -> dict[str, Any]:
    """Set precision / recall / F1 over the returned list, plus MRR and nDCG@k.

    Precision@10 / Recall@10 are deliberately NOT used: HireNex returns the full
    filtered set, and the benchmark queries have between 4 and 31 relevant
    candidates. With 4 relevant candidates a perfect system scores P@10 = 0.4;
    with 31 it scores R@10 = 0.32. Set-level metrics do not have that ceiling.
    """
    predicted = [str(x).upper() for x in predicted]
    expected_set = {str(x).upper() for x in expected}
    returned = set(predicted)
    hits = len(returned & expected_set)
    precision = hits / len(returned) if returned else 0.0
    recall = hits / len(expected_set) if expected_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    reciprocal_rank = next((1.0 / rank for rank, app in enumerate(predicted, start=1) if app in expected_set), 0.0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": reciprocal_rank,
        "ndcg": binary_ndcg(predicted, expected_set, k),
        "relevant": len(expected_set),
        "returned": len(returned),
    }


def benchmark_retrieval(run_search) -> dict[str, Any]:
    queries = load_json(QUERIES_PATH, [])
    rows = []
    models: dict[str, int] = {}
    started_all = time.perf_counter()
    for item in queries:
        query = str(item.get("query", ""))
        expected = item.get("expected_application_nos", [])
        started = time.perf_counter()
        try:
            result = run_search(query)
            candidates = result.get("candidates", []) if isinstance(result, dict) else []
            predicted = [str(c.get("application_no", "")).upper() for c in candidates if isinstance(c, dict)]
            row = {"query": query, "status": "success", **retrieval_metrics(predicted, expected)}
            query_llm = ((result or {}).get("llm") or {}).get("query") or {}
            if query_llm.get("provider"):
                label = f"{query_llm['provider']}:{query_llm.get('model', '')}"
                models[label] = models.get(label, 0) + 1
        except Exception as exc:
            row = {
                "query": query, "status": "failed", "error": str(exc),
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "mrr": 0.0, "ndcg": 0.0,
                "relevant": len(expected), "returned": 0,
            }
        row["latency_seconds"] = round(time.perf_counter() - started, 3)
        rows.append(row)

    ok = [r for r in rows if r["status"] == "success"]

    def mean(key: str) -> float:
        return sum(r[key] for r in ok) / len(ok) if ok else 0.0

    aggregate = {
        "available": bool(ok),
        "precision": mean("precision"),
        "recall": mean("recall"),
        "f1": mean("f1"),
        "mrr": mean("mrr"),
        "ndcg_at_10": mean("ndcg"),
        "avg_latency_seconds": sum(r["latency_seconds"] for r in rows) / len(rows) if rows else 0.0,
        "queries_total": len(rows),
        "queries_successful": len(ok),
        "models_used": models,
        "error_summary": rows[0].get("error", "") if rows and not ok else "",
    }
    return {"aggregate": aggregate, "queries": rows}


def conversion_metrics() -> dict[str, Any]:
    manifest = load_json(MANIFEST, [])
    total = len(manifest)
    converted = sum(1 for r in manifest if r.get("markdown_status") == "success")
    failed = sum(1 for r in manifest if r.get("markdown_status") == "failed")
    times = [float(r["markdown_conversion_seconds"]) for r in manifest if r.get("markdown_conversion_seconds") is not None]
    return {
        "total": total,
        "converted": converted,
        "failed": failed,
        "success_rate": round(converted / total, 4) if total else 0.0,
        "avg_seconds": round(sum(times) / len(times), 3) if times else None,
    }


def run_full_evaluation(run_search=None, include_retrieval: bool = True) -> dict[str, Any]:
    ground_truth = load_json(GROUND_TRUTH_PATH, [])
    if not ground_truth:
        raise FileNotFoundError(f"Ground truth not found: {GROUND_TRUTH_PATH}")

    result: dict[str, Any] = {
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": {
            "ground_truth_records": len(ground_truth),
            "recruiter_queries": len(load_json(QUERIES_PATH, [])),
        },
        "conversion": conversion_metrics(),
        "parsers": {},
        "retrieval": None,
    }

    if NLP_JSON.exists():
        result["parsers"]["nlp"] = evaluate_parser(NLP_JSON, "Traditional NLP", ground_truth)
    if LLM_JSON.exists():
        result["parsers"]["llm"] = evaluate_parser(LLM_JSON, "LLM", ground_truth)

    if include_retrieval and run_search is not None and (NLP_JSON.exists() or LLM_JSON.exists()):
        result["retrieval"] = benchmark_retrieval(run_search)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def load_latest_evaluation() -> dict[str, Any] | None:
    data = load_json(RESULT_PATH, None)
    return data if isinstance(data, dict) else None
