"""Shared client for LLM calls.

Requests JSON and validates it into a pydantic model rather than relying on provider-side
schema enforcement, which is uneven across models: JSON mode guarantees parseable output but
not that it matches a schema, and strict schema support is endpoint-specific.

Handles the failure modes thinking models introduce. A model whose reasoning consumes the
whole token budget returns empty content with a truncated finish reason, so the budget is
raised and the call retried. Reasoning also leaks prose and code fences around the JSON, so
extraction tolerates both.

Responses are cached on disk by content, keyed on everything that can change the answer, so
repeated runs and resumed sweeps do not re-bill for identical calls.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, TypeVar

from pydantic import BaseModel

from dctax.config import REPO_ROOT, data_root

M = TypeVar("M", bound=BaseModel)

_FENCE_OPEN = re.compile(r"^```[a-zA-Z]*\n?")
_FENCE_CLOSE = re.compile(r"\n?```$")
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)

_local = threading.local()
_cache_lock = threading.Lock()


@dataclass(frozen=True)
class LLMConfig:
    """Model and call settings.

    model            provider-qualified model id
    base_url         OpenAI-compatible endpoint
    api_key_env      environment variable holding the key for that endpoint
    temperature      sampling temperature; None leaves the provider's recommended default
    top_p            nucleus sampling threshold; None leaves the provider's recommended default
    max_tokens       completion budget, inclusive of reasoning tokens
    max_tokens_cap   ceiling when the budget is raised after an empty completion
    concurrency      simultaneous in-flight requests
    max_attempts     attempts per request before giving up
    cache            read and write the on-disk response cache
    """

    # :nitro routes to the highest-throughput providers serving the model. Default routing
    # spread requests across providers of varying reliability, which cost 77 of 665 bucketing
    # calls their six attempts.
    model: str = "deepseek/deepseek-v4-flash:nitro"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int = 16384
    max_tokens_cap: int = 32768
    concurrency: int = 8
    max_attempts: int = 6
    timeout: float = 240.0
    cache: bool = True

    def fingerprint(self) -> dict[str, Any]:
        """The settings that can change a response, for cache keying."""
        return {k: v for k, v in asdict(self).items()
                if k in ("model", "base_url", "temperature", "top_p")}


# --------------------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------------------


def api_key(name: str) -> str:
    """Read an API key from the environment, falling back to the repository .env file."""
    key = os.environ.get(name)
    if not key:
        env_file = REPO_ROOT / ".env"
        if env_file.exists():
            from dotenv import dotenv_values

            key = dotenv_values(env_file).get(name)
    if not key:
        raise RuntimeError(f"{name} is not set in the environment or {REPO_ROOT}/.env")
    return key


def _client(cfg: LLMConfig):
    """One client per thread; the OpenAI client is not documented as thread-safe."""
    cached = getattr(_local, "client", None)
    if cached is not None and _local.client_key == (cfg.base_url, cfg.api_key_env):
        return cached
    from openai import OpenAI

    _local.client = OpenAI(base_url=cfg.base_url, api_key=api_key(cfg.api_key_env),
                           timeout=cfg.timeout)
    _local.client_key = (cfg.base_url, cfg.api_key_env)
    return _local.client


# --------------------------------------------------------------------------------------
# Response cache
# --------------------------------------------------------------------------------------


def cache_dir() -> Path:
    return data_root() / "llm_cache"


def _cache_key(cfg: LLMConfig, system: str, payload: Any, schema: type[BaseModel] | None) -> str:
    blob = json.dumps(
        {
            "cfg": cfg.fingerprint(),
            "system": system,
            "payload": payload,
            "schema": sorted(schema.model_fields) if schema else None,
        },
        sort_keys=True,
        default=str,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return cache_dir() / key[:2] / f"{key}.json"


def _cache_read(key: str) -> dict | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _cache_write(key: str, value: dict) -> None:
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with _cache_lock:
        tmp.write_text(json.dumps(value, ensure_ascii=False))
        tmp.replace(path)


# --------------------------------------------------------------------------------------
# JSON extraction and retry
# --------------------------------------------------------------------------------------


def extract_json(text: str) -> dict:
    """Parse a JSON object out of a completion that may carry fences or surrounding prose."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", text)).strip()
    try:
        return json.loads(text)
    except Exception:
        match = _JSON_OBJECT.search(text)
        if match:
            return json.loads(match.group(0))
        raise


def _retry_after(exc: Exception) -> float | None:
    """Seconds requested by a rate-limited response, when it says so."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _backoff(attempt: int, exc: Exception) -> float:
    """Delay before the next attempt, honouring the server's own request when present."""
    requested = _retry_after(exc)
    if requested is not None:
        return min(requested, 60.0)
    return min(2.0 ** attempt, 30.0) * random.uniform(0.5, 1.5)


def _is_rate_limit(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429 or "429" in str(exc)


# --------------------------------------------------------------------------------------
# Calls
# --------------------------------------------------------------------------------------


def structured(
    payload: Any,
    *,
    system: str,
    schema: type[M],
    cfg: LLMConfig | None = None,
) -> M:
    """Send `payload` and validate the reply into `schema`.

    Raises the last error if every attempt fails.
    """
    cfg = cfg or LLMConfig()
    key = _cache_key(cfg, system, payload, schema)
    if cfg.cache:
        hit = _cache_read(key)
        if hit is not None:
            return schema(**hit)

    fields = ", ".join(
        f'"{name}": <{getattr(f.annotation, "__name__", f.annotation)}>'
        for name, f in schema.model_fields.items()
    )
    messages = [
        {"role": "system",
         "content": f"{system}\n\nReturn ONLY a JSON object of the form {{{fields}}} "
                    f"with no other text."},
        {"role": "user",
         "content": json.dumps({"input": payload}, default=str, ensure_ascii=False)},
    ]

    budget = cfg.max_tokens
    json_mode = True
    last: Exception | None = None

    for attempt in range(cfg.max_attempts):
        try:
            # Sampling settings are omitted unless set, so the provider applies the model's
            # own recommended values rather than ours.
            request = dict(model=cfg.model, messages=messages, max_tokens=budget)
            if cfg.temperature is not None:
                request["temperature"] = cfg.temperature
            if cfg.top_p is not None:
                request["top_p"] = cfg.top_p
            if json_mode:
                request["response_format"] = {"type": "json_object"}
            reply = _client(cfg).chat.completions.create(**request)
            choice = reply.choices[0]
            if not (choice.message.content or "").strip():
                raise ValueError(f"empty completion (finish_reason={choice.finish_reason})")
            parsed = schema(**extract_json(choice.message.content))
            if cfg.cache:
                _cache_write(key, parsed.model_dump())
            return parsed
        except Exception as exc:  # noqa: BLE001: every failure mode here is retryable
            last = exc
            if "response_format" in str(exc).lower():
                json_mode = False
            else:
                budget = min(budget * 2, cfg.max_tokens_cap)
            if attempt == cfg.max_attempts - 1:
                break
            time.sleep(_backoff(attempt, exc))

    raise RuntimeError(f"{cfg.model}: {cfg.max_attempts} attempts failed") from last


def map_structured(
    payloads: Sequence[Any],
    *,
    system: str,
    schema: type[M],
    cfg: LLMConfig | None = None,
    progress: bool = True,
) -> list[M | None]:
    """Run `structured` over `payloads` concurrently, preserving input order.

    A payload that exhausts its attempts yields None rather than ending the batch. The first
    failure of each kind is printed with its traceback.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cfg = cfg or LLMConfig()
    results: list[M | None] = [None] * len(payloads)
    reported: set[str] = set()

    def run(index: int):
        return index, structured(payloads[index], system=system, schema=schema, cfg=cfg)

    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        futures = [pool.submit(run, i) for i in range(len(payloads))]
        stream: Iterable = as_completed(futures)
        if progress:
            from tqdm import tqdm

            stream = tqdm(stream, total=len(futures), desc=cfg.model)
        for future in stream:
            try:
                index, value = future.result()
                results[index] = value
            except Exception as exc:  # noqa: BLE001
                name = type(exc).__name__
                if name not in reported:
                    reported.add(name)
                    print(traceback.format_exc())

    failed = sum(r is None for r in results)
    if failed:
        print(f"{failed}/{len(payloads)} payloads failed")
    return results
