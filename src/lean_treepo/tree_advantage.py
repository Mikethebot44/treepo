from __future__ import annotations

from collections import defaultdict
from typing import Any


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = variance ** 0.5
    if std < 1e-8:
        return [0.0 for _ in values]
    return [(value - mean) / (std + 1e-4) for value in values]


def group_grpo_advantages(rewards: list[float], num_generations: int) -> list[float]:
    output: list[float] = []
    for start in range(0, len(rewards), num_generations):
        output.extend(normalize(rewards[start : start + num_generations]))
    return output[: len(rewards)]


def tree_token_advantages(
    rewards: list[float],
    tree_metadata: list[dict[str, Any]],
    completion_lengths: list[int],
    num_generations: int,
) -> list[list[float]]:
    fallback = group_grpo_advantages(rewards, num_generations)
    result: list[list[float]] = [[fallback[idx]] * completion_lengths[idx] for idx in range(len(rewards))]

    subgroup: dict[tuple[int, str, int], list[int]] = defaultdict(list)
    for rollout_idx, metadata in enumerate(tree_metadata):
        for segment in metadata.get("segments", []):
            key = (
                int(metadata.get("prompt_index", rollout_idx // max(1, num_generations))),
                str(segment.get("parent_id", "")),
                int(segment.get("depth", 0)),
            )
            subgroup[key].append(rollout_idx)

    for indices in subgroup.values():
        unique_indices = sorted(set(indices))
        if len(unique_indices) < 2:
            continue
        values = normalize([rewards[idx] for idx in unique_indices])
        for idx, advantage in zip(unique_indices, values, strict=True):
            for segment in tree_metadata[idx].get("segments", []):
                start = max(0, int(segment.get("segment_start", 0)))
                end = min(completion_lengths[idx], int(segment.get("segment_end", completion_lengths[idx])))
                for token_idx in range(start, end):
                    result[idx][token_idx] = advantage
    return result
