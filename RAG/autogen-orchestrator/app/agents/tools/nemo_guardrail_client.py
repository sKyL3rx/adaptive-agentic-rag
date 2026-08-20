from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests

from config import settings_for_explainer_agent

logger = logging.getLogger(__name__)

DEFAULT_MODEL = (
    getattr(settings_for_explainer_agent, "openai_model", None)
    or "Qwen/Qwen2.5-7B-Instruct"
)

SAFE_FALLBACK = (
    "Sorry, I can’t help with that content. "
    "If you have another safe and appropriate question, I’m happy to help."
)

REQUEST_TIMEOUT_SECONDS = 30


def _api_root() -> str:
    """
    Return NeMo Guardrails base URL, e.g. http://localhost:7331
    """
    base_url = (settings_for_explainer_agent.nemo_url or "").strip()
    if not base_url:
        raise RuntimeError("NEMO_URL is not set (check your .env or environment)")
    return base_url.rstrip("/")


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST JSON to NeMo Guardrails and return decoded JSON.
    """
    url = f"{_api_root()}{path}"
    response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _pick(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _dig(obj: dict, path: list[str], default=None):
    current = obj
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _guardrails_options() -> Dict[str, Any]:
    return {
        "output_vars": True,
        "log": {
            "activated_rails": True,
            "llm_calls": True,
        },
    }


def _guardrails_block(config_id: str) -> Dict[str, Any]:
    return {
        "config_id": config_id,
        "options": _guardrails_options(),
    }


def guard_in(user_text: str) -> Dict[str, Any]:
    """
    Run INPUT rails only.
    Endpoint: POST {NEMO_URL}/v1/guardrail/checks
    """
    config_id = settings_for_explainer_agent.nemo_config_in or "nemoguard_v9_in"

    body = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": user_text}],
        "guardrails": _guardrails_block(config_id),
    }
    return _post("/v1/guardrail/checks", body)


def guard_out(
    assistant_text: str,
    *,
    user_text: Optional[str] = None,
    config_id: Optional[str] = None,
    include_raw: bool = False,
    fail_close: bool = False,
) -> Dict[str, Any]:
    """
    Run OUTPUT rails only.

    Returns a normalized verdict:
      {
        "passed": bool,
        "final_response": Optional[str],
        "status": Optional[str],
        "activated_rails": Optional[list],
        "violations": Optional[list],
        "raw": Optional[dict],   # only if include_raw=True
      }
    """
    resolved_config_id = (
        config_id
        or getattr(settings_for_explainer_agent, "nemo_config_out", None)
        or "nemoguard_v9_out"
    )

    body = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": user_text or ""},
            {"role": "assistant", "content": assistant_text},
        ],
        "guardrails": _guardrails_block(resolved_config_id),
    }

    try:
        raw = _post("/v1/guardrail/checks", body)
    except Exception as exc:
        logger.error("NeMo guard_out failed: %s", exc)

        if fail_close:
            result: Dict[str, Any] = {
                "passed": False,
                "final_response": "Sorry, I can’t help with that content.",
                "status": "error",
                "activated_rails": None,
                "violations": None,
            }
        else:
            result = {
                "passed": True,
                "final_response": None,
                "status": "error",
                "activated_rails": None,
                "violations": None,
            }

        if include_raw:
            result["raw"] = {"error": str(exc)}
        return result

    status = _pick(
        raw.get("status"),
        _dig(raw, ["result", "status"]),
        "allowed"
        if _pick(
            raw.get("allowed"),
            _dig(raw, ["result", "allowed"]),
            _dig(raw, ["guardrails_data", "output_data", "allowed"]),
            _dig(raw, ["response", "allowed"]),
        )
        is True
        else None,
        "blocked"
        if _pick(
            raw.get("allowed"),
            _dig(raw, ["result", "allowed"]),
            _dig(raw, ["guardrails_data", "output_data", "allowed"]),
            _dig(raw, ["response", "allowed"]),
        )
        is False
        else None,
        _dig(raw, ["rails_status", "content safety check output $model=content_safety", "status"]),
    )

    passed = _pick(
        raw.get("passed"),
        _dig(raw, ["result", "passed"]),
    )

    if passed is None:
        if isinstance(status, str):
            normalized_status = status.lower()
            if normalized_status in {
                "allowed",
                "ok",
                "pass",
                "success",
                "allow",
                "allowed_with_modifications",
            }:
                passed = True
            elif normalized_status in {
                "blocked",
                "block",
                "deny",
                "denied",
                "rejected",
                "unsafe",
                "error",
            }:
                passed = False

    if passed is None:
        allowed = _pick(
            raw.get("allowed"),
            _dig(raw, ["result", "allowed"]),
            _dig(raw, ["guardrails_data", "output_data", "allowed"]),
            _dig(raw, ["response", "allowed"]),
        )
        if allowed is not None:
            passed = bool(allowed)

    if passed is None:
        passed = True

    final_response = _pick(
        raw.get("final_response"),
        _dig(raw, ["output_vars", "final_response"]),
        _dig(raw, ["guardrails_data", "output_data", "final_response"]),
        _dig(raw, ["response", "content"]) if isinstance(raw.get("response"), dict) else None,
        raw.get("message"),
    )

    activated_rails_detail = _pick(
        raw.get("activated_rails"),
        _dig(raw, ["result", "activated_rails"]),
        _dig(raw, ["log", "activated_rails"]),
        _dig(raw, ["guardrails_data", "log", "activated_rails"]),
    )

    violations_detail = _pick(
        raw.get("violations"),
        raw.get("violations_detected"),
        _dig(raw, ["guardrails_data", "output_data", "policy_violations"]),
        _dig(raw, ["response", "policy_violations"]),
    )

    if status is None:
        status = "allowed" if passed else "blocked"

    result = {
        "passed": bool(passed),
        "final_response": final_response,
        "status": status,
        "activated_rails": activated_rails_detail if include_raw else None,
        "violations": violations_detail if include_raw else None,
    }

    if include_raw:
        result["raw"] = raw

    return result


def inline_guard_body() -> Dict[str, Any]:
    """
    Helper for /v1/guardrail/chat/completions if you want inline NeMo pre/post rails.
    """
    return {
        "guardrails": {
            "config_id": settings_for_explainer_agent.nemo_config_inline or "nemoguard_v9_out"
        },
        "options": _guardrails_options(),
    }