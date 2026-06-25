from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class LeanVerification:
    success: bool
    feedback: str
    raw: dict[str, Any]
    latency_seconds: float = 0.0


class KiminaLeanClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout_seconds: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def verify(self, proof: str, custom_id: str = "candidate") -> LeanVerification:
        payload = {"codes": [{"custom_id": custom_id, "proof": proof}], "infotree_type": "original"}
        started = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/verify",
            headers=self.headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        latency = time.perf_counter() - started
        response.raise_for_status()
        parsed = parse_kimina_response(response.json())
        return LeanVerification(parsed.success, parsed.feedback, parsed.raw, latency)


def parse_kimina_response(data: dict[str, Any]) -> LeanVerification:
    entries = _find_entries(data)
    if not entries:
        return LeanVerification(False, f"Kimina returned no result entries: {data}", data)
    entry = entries[0]
    success = _entry_success(entry)
    feedback = _entry_feedback(entry, success)
    return LeanVerification(success, feedback, data)


def _find_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("results", "data", "outputs", "codes"):
        value = data.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value
    if any(key in data for key in ("custom_id", "messages", "errors", "diagnostics", "status")):
        return [data]
    return []


def _entry_success(entry: dict[str, Any]) -> bool:
    for key in ("success", "verified", "is_valid", "valid", "passed"):
        if isinstance(entry.get(key), bool):
            return bool(entry[key])
    status = entry.get("status")
    if isinstance(status, str):
        return status.lower() in {"ok", "success", "valid", "verified", "passed", "complete"}
    if entry.get("error") or entry.get("errors") or entry.get("diagnostics") or entry.get("messages"):
        return False
    return False


def _entry_feedback(entry: dict[str, Any], success: bool) -> str:
    for key in ("feedback", "error", "stderr", "message", "diagnostics", "messages", "errors"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value:
            return str(value)
    return "Lean verification succeeded." if success else f"Lean verification failed: {entry}"
