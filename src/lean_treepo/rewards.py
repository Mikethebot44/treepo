from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .formatting import extract_lean_from_completion
from .lean_client import KiminaLeanClient


@dataclass(frozen=True)
class RewardTrace:
    reward: float
    extraction_mode: str
    verified: bool
    feedback: str
    latency_seconds: float


class LeanVerifierReward:
    def __init__(self, client: KiminaLeanClient, max_workers: int = 8) -> None:
        self.client = client
        self.max_workers = max_workers
        self.last_traces: list[RewardTrace] = []

    def _score_one(self, idx: int, completion: Any, formal_statement: str | None) -> RewardTrace:
        extracted = extract_lean_from_completion(completion, formal_statement=formal_statement)
        if not extracted.ok:
            return RewardTrace(0.0, extracted.mode, False, "No Lean code extracted.", 0.0)
        try:
            result = self.client.verify(extracted.code, custom_id=f"lean-treepo-{idx}")
        except Exception as exc:  # noqa: BLE001 - remote verifier failures are reward failures.
            return RewardTrace(0.0, extracted.mode, False, f"Kimina request failed: {exc}", 0.0)
        return RewardTrace(
            reward=1.0 if result.success else 0.0,
            extraction_mode=extracted.mode,
            verified=result.success,
            feedback=result.feedback,
            latency_seconds=result.latency_seconds,
        )

    def __call__(self, completions: list[Any], formal_statement: list[str] | None = None, log_extra=None, log_metric=None, **_: Any) -> list[float]:
        statements: list[str | None]
        if not formal_statement:
            statements = [None] * len(completions)
        elif len(formal_statement) == len(completions):
            statements = list(formal_statement)
        elif len(completions) % len(formal_statement) == 0:
            repeats = len(completions) // len(formal_statement)
            statements = [statement for statement in formal_statement for _ in range(repeats)]
        else:
            statements = list(formal_statement) + [formal_statement[-1]] * (len(completions) - len(formal_statement))
        tasks = [(idx, completion, statement) for idx, (completion, statement) in enumerate(zip(completions, statements, strict=False))]
        with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as executor:
            traces = list(executor.map(lambda item: self._score_one(item[0], item[1], item[2]), tasks))
        self.last_traces = traces

        rewards = [trace.reward for trace in traces]
        if log_extra:
            log_extra("lean_extraction_mode", [trace.extraction_mode for trace in traces])
            log_extra("lean_feedback", [trace.feedback for trace in traces])
        if log_metric and traces:
            log_metric("lean/accuracy", sum(rewards) / len(rewards))
            log_metric("lean/extraction_rate", sum(trace.extraction_mode != "no_lean_found" for trace in traces) / len(traces))
            log_metric("lean/mean_verify_latency", sum(trace.latency_seconds for trace in traces) / len(traces))
        return rewards
