"""
Implementation of safety checks using Guardrails.  
"""
from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx


GUARDRAILS_URL = os.getenv(
    "GUARDRAILS_URL",
    "http://localhost:7331/v1/guardrail/checks",
)
GUARDRAILS_TIMEOUT_SECONDS = float(os.getenv("GUARDRAILS_TIMEOUT_SECONDS", "6.0"))
GUARDRAILS_FAIL_OPEN = os.getenv("GUARDRAILS_FAIL_OPEN", "true").lower() == "true"
GUARDRAILS_CONFIG_ID = os.getenv("GUARDRAILS_CONFIG_ID", "default/nemoguard_v9_in") 




@dataclass
class SafetyDecision:
    blocked: bool
    flow: str


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _build_nemo_payload(user_text: str) -> dict[str, Any]:
    return {
        "model": "nemo-guardrails",
        "messages": [
            {
                "role": "user",
                "content": user_text,
            }
        ],
        "guardrails": {
            "config_id": GUARDRAILS_CONFIG_ID,
        },

    }


def _extract_block_from_nemo_response(data: dict[str, Any]) -> bool:
    """
    Extracts the overall block status from the Nemo Guardrails response.
    """
    overall_status = data.get("status")

    if overall_status == "blocked":
        return True
    else:
        return False


async def nemo_guard_check(user_text: str) -> SafetyDecision:
    payload = _build_nemo_payload(user_text)

    try:
        async with httpx.AsyncClient(timeout=GUARDRAILS_TIMEOUT_SECONDS) as client:
            response = await client.post(GUARDRAILS_URL, json=payload)
            response.raise_for_status()
            data = response.json()

        blocked = _extract_block_from_nemo_response(data)

        return SafetyDecision(
            blocked=blocked,
            flow="Nemo passed" if not blocked else "Nemo blocked",
        )

    except Exception:
        if GUARDRAILS_FAIL_OPEN:
            return SafetyDecision(
                blocked=False,
                flow="Nemo error (fail open)",
            )

        return SafetyDecision(
            blocked=True,
            flow="Nemo error (fail close)",
        )


async def run_input_guardrails(user_text: str) -> SafetyDecision:
    
    nemo_decision = await nemo_guard_check(_normalize(user_text))

    return nemo_decision
