"""
HireNex - AI provider layer (Gemini + Groq only)

Every LLM call in HireNex goes through this module, so there is exactly one
place that knows about API keys, models, routing, retries and fallbacks.

Task routing (why each provider is the default for its job)
-----------------------------------------------------------
parsing  -> Gemini first, Groq as fallback
            Resume -> nested JSON is a long, schema-heavy task. Gemini supports
            native schema-constrained output and a very large context window.
query    -> Groq first, Gemini as fallback
answer   -> Groq first, Gemini as fallback
            Retrieval makes two short, latency-sensitive calls per search
            (query understanding + grounded answer). Groq's inference speed
            matters most here, and it keeps retrieval traffic off the Gemini
            quota that resume parsing uses.

If the preferred provider has no key, is rate limited, or the model has been
retired, the other provider is used automatically. With only ONE key
configured, that provider serves every task.

Everything is configurable from .env (see .env.example).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv()

PROVIDERS = ("gemini", "groq")
TASKS = ("parsing", "query", "answer")

# Defaults favour stable models currently listed in the provider model catalogs.
# All model IDs remain configurable through .env because account access can vary.
DEFAULTS = {
    "GEMINI_MODEL": "gemini-3.8-flash",
    "GEMINI_FALLBACK_MODEL": "gemini-3.5-flash-lite",
    "GROQ_PARSER_MODEL": "openai/gpt-oss-120b",
    "GROQ_QUERY_MODEL": "openai/gpt-oss-120b",
    "GROQ_ANSWER_MODEL": "openai/gpt-oss-120b",
    "GROQ_FALLBACK_MODEL": "openai/gpt-oss-20b",
    "PARSER_PROVIDER": "gemini",
    "RETRIEVAL_PROVIDER": "groq",
}

MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))
MAX_WAIT_SECONDS = float(os.getenv("LLM_MAX_WAIT_SECONDS", "300"))


def _env(name: str) -> str:
    return os.getenv(name, DEFAULTS.get(name, "")).strip()


def _api_key(provider: str) -> str:
    name = "GEMINI_API_KEY" if provider == "gemini" else "GROQ_API_KEY"
    key = os.getenv(name, "").strip().strip("\"'")
    # .env.example ships with placeholders such as "your_gemini_key".
    if not key or key.lower().startswith("your_"):
        return ""
    return key


def configured_providers() -> list[str]:
    return [p for p in PROVIDERS if _api_key(p)]


# -----------------------------------------------------------------------------
# Routing
# -----------------------------------------------------------------------------

def preferred_provider(task: str) -> str:
    name = "PARSER_PROVIDER" if task == "parsing" else "RETRIEVAL_PROVIDER"
    value = _env(name).lower()
    return value if value in PROVIDERS else DEFAULTS[name]


def provider_order(task: str) -> list[str]:
    """Preferred provider first, the other as fallback; keyless ones removed."""
    first = preferred_provider(task)
    other = "groq" if first == "gemini" else "gemini"
    return [p for p in (first, other) if _api_key(p)]


def model_chain(provider: str, task: str) -> list[str]:
    """Primary model for the task, then the fallback model (deduplicated)."""
    if provider == "gemini":
        models = [_env("GEMINI_MODEL"), _env("GEMINI_FALLBACK_MODEL")]
    else:
        primary = {
            "parsing": "GROQ_PARSER_MODEL",
            "query": "GROQ_QUERY_MODEL",
            "answer": "GROQ_ANSWER_MODEL",
        }[task]
        models = [_env(primary), _env("GROQ_FALLBACK_MODEL")]
    seen: list[str] = []
    for model in models:
        if model and model not in seen:
            seen.append(model)
    return seen


# -----------------------------------------------------------------------------
# JSON helper
# -----------------------------------------------------------------------------

def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply (tolerates fences)."""
    cleaned = re.sub(r"```(?:json)?", "", text or "", flags=re.I).strip()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("Model reply did not contain a JSON object.")
    obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(obj, dict):
        raise ValueError("Model reply JSON was not an object.")
    return obj


# -----------------------------------------------------------------------------
# Result / errors
# -----------------------------------------------------------------------------

@dataclass
class LLMMeta:
    provider: str = ""
    model: str = ""
    seconds: float = 0.0
    fallback_used: bool = False
    errors: list[str] = field(default_factory=list)

    def label(self) -> str:
        return f"{self.provider}:{self.model}" if self.provider else ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "seconds": round(self.seconds, 3),
            "fallback_used": self.fallback_used,
        }


class LLMUnavailableError(RuntimeError):
    """Raised when no configured provider/model could complete the request."""


class _SkipCandidate(Exception):
    """Internal: give up on this provider/model and try the next one."""

    def __init__(self, reason: str, *, dead: bool = False, cooldown: float = 0.0, provider_dead: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.dead = dead
        self.cooldown = cooldown
        self.provider_dead = provider_dead


# -----------------------------------------------------------------------------
# Error classification
# -----------------------------------------------------------------------------

def _status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _retry_hint(exc: Exception) -> float | None:
    """Seconds the API asked us to wait, if the message or headers say so."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        for key in ("retry-after", "Retry-After"):
            try:
                if headers.get(key):
                    return float(headers.get(key))
            except (TypeError, ValueError):
                pass
    text = str(exc)
    match = re.search(r"retry(?:Delay)?\D{0,15}?(\d+(?:\.\d+)?)\s*(ms|s)\b", text, re.I)
    if not match:
        match = re.search(r"try again in\s*(?:(\d+)m)?\s*(\d+(?:\.\d+)?)\s*(ms|s)", text, re.I)
        if match:
            minutes = float(match.group(1) or 0)
            seconds = float(match.group(2))
            return minutes * 60 + (seconds / 1000 if match.group(3).lower() == "ms" else seconds)
        return None
    value = float(match.group(1))
    return value / 1000 if match.group(2).lower() == "ms" else value


def _is_transient(exc: Exception) -> bool:
    status = _status_code(exc)
    if status in {408, 409, 429, 500, 502, 503, 504}:
        return True
    return type(exc).__name__ in {"APIConnectionError", "APITimeoutError", "ReadTimeout", "ConnectError", "TimeoutError"}


def _short(exc: Exception) -> str:
    text = re.sub(r"\s+", " ", str(exc)).strip()
    return f"{type(exc).__name__}: {text[:240]}"


# -----------------------------------------------------------------------------
# Provider clients
# -----------------------------------------------------------------------------

_clients: dict[tuple[str, str], Any] = {}
_client_lock = threading.Lock()


def _client(provider: str) -> Any:
    key = _api_key(provider)
    if not key:
        raise LLMUnavailableError(f"{provider.upper()}_API_KEY is not configured.")
    with _client_lock:
        cached = _clients.get((provider, key))
        if cached is not None:
            return cached
        if provider == "gemini":
            from google import genai

            cached = genai.Client(api_key=key)
        else:
            from groq import Groq

            cached = Groq(api_key=key)
        _clients[(provider, key)] = cached
        return cached


# Optional pacing, e.g. GEMINI_MIN_INTERVAL_SECONDS=6 for a 10 requests/minute plan.
_last_call: dict[str, float] = {}
_pace_lock = threading.Lock()


def _pace(provider: str) -> None:
    try:
        interval = float(os.getenv(f"{provider.upper()}_MIN_INTERVAL_SECONDS", "0") or 0)
    except ValueError:
        interval = 0.0
    if interval <= 0:
        return
    with _pace_lock:
        wait = _last_call.get(provider, 0.0) + interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call[provider] = time.monotonic()


def _gemini_thinking(model: str, types: Any) -> Any:
    """Keep extraction fast: little or no reasoning for these simple tasks."""
    name = model.lower()
    try:
        if "2.5-flash-lite" in name or "2.5-pro" in name:
            return None
        if "2.5-flash" in name:
            return types.ThinkingConfig(thinking_budget=0)
        if name.startswith("gemini-3") or name.startswith("gemini-4"):
            return types.ThinkingConfig(thinking_level="low")
    except Exception:
        return None
    return None


def _call_gemini(model: str, system: str, user: str, *, json_mode: bool, schema: Any, temperature: float) -> str:
    from google.genai import types

    kwargs: dict[str, Any] = {"system_instruction": system, "temperature": temperature}
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
        if schema is not None:
            kwargs["response_schema"] = schema
    thinking = _gemini_thinking(model, types)
    if thinking is not None:
        kwargs["thinking_config"] = thinking
    response = _client("gemini").models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(**kwargs),
    )
    text = response.text
    if not text:
        raise ValueError("Gemini returned an empty response.")
    return text


def _call_groq(model: str, system: str, user: str, *, json_mode: bool, temperature: float, reasoning: str) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if model.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = reasoning
    response = _client("groq").chat.completions.create(**kwargs)
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("Groq returned an empty response.")
    return text


# -----------------------------------------------------------------------------
# Dead-model / cool-down memory (so 90 resumes do not each re-discover a 404)
# -----------------------------------------------------------------------------

_dead: set[tuple[str, str]] = set()
_dead_providers: set[str] = set()
_cooldown: dict[tuple[str, str], float] = {}
_state_lock = threading.Lock()


def reset_provider_state() -> None:
    """Forget retired models / cool-downs (used after editing .env)."""
    with _state_lock:
        _dead.clear()
        _dead_providers.clear()
        _cooldown.clear()
    with _client_lock:
        _clients.clear()


def _available(provider: str, model: str) -> bool:
    with _state_lock:
        if provider in _dead_providers or (provider, model) in _dead:
            return False
        return _cooldown.get((provider, model), 0.0) <= time.monotonic()


def _call_with_retry(provider: str, model: str, call: Callable[[], str]) -> str:
    for attempt in range(MAX_RETRIES + 1):
        _pace(provider)
        try:
            return call()
        except ValueError:
            raise
        except LLMUnavailableError:
            raise
        except Exception as exc:  # SDK-specific errors are classified below
            status = _status_code(exc)
            text = str(exc).lower()
            if status in {401, 403}:
                raise _SkipCandidate(_short(exc), provider_dead=True) from exc
            if status == 404 or "no longer available" in text or "model_not_found" in text or "decommissioned" in text:
                raise _SkipCandidate(_short(exc), dead=True) from exc
            if not _is_transient(exc):
                raise _SkipCandidate(_short(exc)) from exc
            hint = _retry_hint(exc)
            if hint is not None and hint > MAX_WAIT_SECONDS:
                raise _SkipCandidate(_short(exc), cooldown=min(hint, 600.0)) from exc
            if attempt >= MAX_RETRIES:
                raise _SkipCandidate(_short(exc), cooldown=30.0) from exc
            time.sleep((hint + 0.5) if hint is not None else min(1.5 * (2 ** attempt), 20.0))
    raise _SkipCandidate("Retries exhausted")  # pragma: no cover


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def _run(
    task: str,
    system: str,
    user: str,
    *,
    json_mode: bool,
    schema: Any = None,
    validator: Callable[[str], Any] | None = None,
    temperature: float = 0.1,
    reasoning: str = "low",
) -> tuple[Any, LLMMeta]:
    if task not in TASKS:
        raise ValueError(f"Unknown LLM task: {task}")
    order = provider_order(task)
    if not order:
        raise LLMUnavailableError(
            "No LLM API key is configured. Add GEMINI_API_KEY and/or GROQ_API_KEY to your .env file."
        )

    meta = LLMMeta()
    errors: list[str] = []
    first_provider = order[0]
    first_model = model_chain(first_provider, task)[0]
    started = time.perf_counter()

    for provider in order:
        for model in model_chain(provider, task):
            if not _available(provider, model):
                continue

            def call(provider=provider, model=model) -> str:
                if provider == "gemini":
                    return _call_gemini(model, system, user, json_mode=json_mode, schema=schema, temperature=temperature)
                return _call_groq(model, system, user, json_mode=json_mode, temperature=temperature, reasoning=reasoning)

            try:
                text = _call_with_retry(provider, model, call)
                result = validator(text) if validator else text
            except _SkipCandidate as skip:
                errors.append(f"{provider}:{model} -> {skip.reason}")
                with _state_lock:
                    if skip.provider_dead:
                        _dead_providers.add(provider)
                    elif skip.dead:
                        _dead.add((provider, model))
                    elif skip.cooldown:
                        _cooldown[(provider, model)] = time.monotonic() + skip.cooldown
                if skip.provider_dead:
                    break
                continue
            except Exception as exc:  # empty reply or invalid / unusable output
                errors.append(f"{provider}:{model} -> {_short(exc)}")
                continue

            meta.provider, meta.model = provider, model
            meta.seconds = time.perf_counter() - started
            meta.fallback_used = (provider, model) != (first_provider, first_model)
            meta.errors = errors
            return result, meta

    raise LLMUnavailableError(
        f"All LLM providers failed for task '{task}'. " + " | ".join(errors[-4:])
        if errors
        else f"No available model for task '{task}' (all are cooling down or retired). Check your model names in .env."
    )


def generate_json(
    task: str,
    system: str,
    user: str,
    *,
    schema: Any = None,
    validator: Callable[[str], Any] | None = None,
    temperature: float = 0.1,
    reasoning: str = "low",
) -> tuple[Any, LLMMeta]:
    """JSON completion. Without a validator the result is the parsed dict."""
    return _run(
        task, system, user,
        json_mode=True, schema=schema,
        validator=validator or extract_json,
        temperature=temperature, reasoning=reasoning,
    )


def generate_text(
    task: str,
    system: str,
    user: str,
    *,
    temperature: float = 0.1,
    reasoning: str = "low",
) -> tuple[str, LLMMeta]:
    text, meta = _run(
        task, system, user,
        json_mode=False, temperature=temperature, reasoning=reasoning,
        validator=lambda value: value.strip(),
    )
    return text, meta


# -----------------------------------------------------------------------------
# Status + live connection test (used by the System page)
# -----------------------------------------------------------------------------

def provider_status() -> dict[str, Any]:
    configured = configured_providers()
    return {
        "configured": configured,
        "providers": {
            "gemini": {
                "configured": "gemini" in configured,
                "models": model_chain("gemini", "parsing"),
            },
            "groq": {
                "configured": "groq" in configured,
                "models": {
                    "parsing": model_chain("groq", "parsing"),
                    "query": model_chain("groq", "query"),
                    "answer": model_chain("groq", "answer"),
                },
            },
        },
        "routing": {
            "parsing": provider_order("parsing"),
            "query": provider_order("query"),
            "answer": provider_order("answer"),
        },
        "preferred": {
            "parsing": preferred_provider("parsing"),
            "retrieval": preferred_provider("query"),
        },
    }


def test_connections() -> list[dict[str, Any]]:
    """Ping every configured provider/model once. No retries, no fallbacks."""
    system = "You are a connectivity check. Reply with the single word OK."
    results: list[dict[str, Any]] = []
    for provider in configured_providers():
        models: list[str] = []
        for task in TASKS:
            for model in model_chain(provider, task):
                if model not in models:
                    models.append(model)
        for model in models:
            started = time.perf_counter()
            row: dict[str, Any] = {"provider": provider, "model": model, "ok": False, "seconds": None, "error": ""}
            try:
                if provider == "gemini":
                    _call_gemini(model, system, "ping", json_mode=False, schema=None, temperature=0)
                else:
                    _call_groq(model, system, "ping", json_mode=False, temperature=0, reasoning="low")
                row["ok"] = True
            except Exception as exc:
                row["error"] = _short(exc)
            row["seconds"] = round(time.perf_counter() - started, 2)
            results.append(row)
    return results


if __name__ == "__main__":
    print(json.dumps(provider_status(), indent=2))
    for row in test_connections():
        mark = "OK  " if row["ok"] else "FAIL"
        print(f"[{mark}] {row['provider']:<7} {row['model']:<30} {row['seconds']}s {row['error']}")
