from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable
import json


@dataclass
class SegmentMeta:
    node_id: str
    parent_id: str
    depth: int
    segment_start: int
    segment_end: int
    segment_logprobs: list[float] = field(default_factory=list)
    finish_reason: str = "unfinished"


@dataclass
class TreeNode:
    node_id: str
    parent_id: str
    prompt_ids: list[int]
    completion_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    segments: list[SegmentMeta] = field(default_factory=list)
    depth: int = 0
    finish_reason: str = "unfinished"

    @property
    def active(self) -> bool:
        return self.finish_reason == "unfinished"


@dataclass(frozen=True)
class TreeSamplingStats:
    active_nodes: int
    fallback_count: int
    pruned_count: int
    mean_depth: float
    rollout_seconds: float
    tokenps: float
    trajps: float


def has_repetition(text: str, ngram_chars: int = 32, count_threshold: int = 8) -> bool:
    if not text or len(text) < ngram_chars:
        return False
    counts: dict[str, int] = {}
    for idx in range(0, len(text) - ngram_chars + 1):
        piece = text[idx : idx + ngram_chars]
        counts[piece] = counts.get(piece, 0) + 1
        if counts[piece] >= count_threshold:
            return True
    return False


def initial_divergence(method: str, fixed_width: int, more_min: int, more_max: int, rng: random.Random) -> int:
    if method == "treepo_more_init":
        return rng.randint(more_min, more_max)
    return fixed_width


def compute_tree_stats(leaves: list[TreeNode], fallback_count: int, pruned_count: int, seconds: float) -> TreeSamplingStats:
    total_tokens = sum(len(node.completion_ids) for node in leaves)
    mean_depth = sum(node.depth for node in leaves) / len(leaves) if leaves else 0.0
    return TreeSamplingStats(
        active_nodes=sum(1 for node in leaves if node.active),
        fallback_count=fallback_count,
        pruned_count=pruned_count,
        mean_depth=mean_depth,
        rollout_seconds=seconds,
        tokenps=total_tokens / seconds if seconds > 0 else 0.0,
        trajps=len(leaves) / seconds if seconds > 0 else 0.0,
    )


def _tokenize_prompt(prompt: Any, tokenizer: Any) -> list[int]:
    if isinstance(prompt, list) and hasattr(tokenizer, "apply_chat_template"):
        ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        return list(ids)
    text = str(prompt)
    return list(tokenizer(text, add_special_tokens=True)["input_ids"])


def _is_eos(ids: list[int], tokenizer: Any) -> bool:
    if not ids:
        return True
    eos_ids = {getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "pad_token_id", None)}
    return ids[-1] in eos_ids


def _clone_for_branch(parent: TreeNode, child_id: str) -> TreeNode:
    return TreeNode(
        node_id=child_id,
        parent_id=parent.node_id,
        prompt_ids=list(parent.prompt_ids),
        completion_ids=list(parent.completion_ids),
        logprobs=list(parent.logprobs),
        segments=list(parent.segments),
        depth=parent.depth,
        finish_reason="unfinished",
    )


def _decode_completion(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True) if token_ids else ""


def _prompt_key(prompt: Any) -> str:
    try:
        return json.dumps(prompt, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return repr(prompt)


def _generate_segment_hf(
    trainer: Any,
    input_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[list[int], list[float]]:
    import torch

    tokenizer = trainer.processing_class
    model = trainer.accelerator.unwrap_model(trainer.model)
    device = trainer.accelerator.device
    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(tensor)
    with torch.no_grad():
        output = model.generate(
            input_ids=tensor,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else 0,
            max_new_tokens=max_new_tokens,
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            return_dict_in_generate=True,
            output_scores=True,
        )
    sequence = output.sequences[0].tolist()
    new_ids = sequence[len(input_ids) :]
    logprobs: list[float] = []
    for idx, scores in enumerate(output.scores or []):
        if idx >= len(new_ids):
            break
        logprob = torch.log_softmax(scores[0], dim=-1)[new_ids[idx]].item()
        logprobs.append(float(logprob))
    return new_ids, logprobs


def sample_tree_for_prompt(
    prompt_ids: list[int],
    tokenizer: Any,
    generate_segment: Callable[[list[int]], tuple[list[int], list[float]]],
    method: str,
    max_width: int,
    max_depth: int,
    segment_length: int,
    fixed_step_width: int,
    more_init_min: int,
    more_init_max: int,
    repetition_ngram_chars: int,
    repetition_count: int,
    rng: random.Random,
) -> tuple[list[TreeNode], TreeSamplingStats]:
    started = time.perf_counter()
    root = TreeNode(node_id="0", parent_id="", prompt_ids=list(prompt_ids))
    active = [root]
    leaves: list[TreeNode] = []
    fallback_count = 0
    pruned_count = 0
    infer_step = 0

    while active and len(leaves) < max_width:
        next_active: list[TreeNode] = []
        branch_width = initial_divergence(method, fixed_step_width, more_init_min, more_init_max, rng) if infer_step == 0 else fixed_step_width
        branch_width = max(1, min(branch_width, max_width - len(leaves)))

        for parent_index, parent in enumerate(active):
            if len(leaves) >= max_width:
                break
            for branch_index in range(branch_width):
                if len(leaves) + len(next_active) >= max_width and next_active:
                    break
                child = _clone_for_branch(parent, f"{parent.node_id}/{infer_step}-{parent_index}-{branch_index}")
                segment_ids, segment_logprobs = generate_segment(child.prompt_ids + child.completion_ids)
                start = len(child.completion_ids)
                child.completion_ids.extend(segment_ids)
                child.logprobs.extend(segment_logprobs)
                child.depth += 1
                decoded = _decode_completion(tokenizer, segment_ids)
                finish_reason = "unfinished"
                if _is_eos(segment_ids, tokenizer):
                    finish_reason = "eos"
                elif len(segment_ids) < segment_length:
                    finish_reason = "short_segment"
                elif child.depth >= max_depth:
                    finish_reason = "max_depth"
                elif has_repetition(decoded, repetition_ngram_chars, repetition_count):
                    finish_reason = "repetition"
                    pruned_count += 1

                end = len(child.completion_ids)
                child.segments.append(
                    SegmentMeta(
                        node_id=child.node_id,
                        parent_id=parent.node_id,
                        depth=child.depth,
                        segment_start=start,
                        segment_end=end,
                        segment_logprobs=segment_logprobs,
                        finish_reason=finish_reason,
                    )
                )
                child.finish_reason = finish_reason
                if finish_reason == "unfinished":
                    next_active.append(child)
                else:
                    leaves.append(child)

        active = next_active
        if not active and len(leaves) < max_width and leaves:
            fallback_count += 1
            fallback_source = rng.choice(leaves)
            active = [
                TreeNode(
                    node_id=f"{fallback_source.node_id}/fallback-{fallback_count}",
                    parent_id=fallback_source.parent_id,
                    prompt_ids=list(fallback_source.prompt_ids),
                    completion_ids=list(fallback_source.completion_ids[: max(0, len(fallback_source.completion_ids) - segment_length)]),
                    logprobs=list(fallback_source.logprobs[: max(0, len(fallback_source.logprobs) - segment_length)]),
                    segments=list(fallback_source.segments[:-1]),
                    depth=max(0, fallback_source.depth - 1),
                    finish_reason="unfinished",
                )
            ]
        infer_step += 1

    leaves.extend(active)
    leaves = leaves[:max_width]
    while len(leaves) < max_width and leaves:
        fallback_count += 1
        leaves.append(_clone_for_branch(rng.choice(leaves), f"fallback-dup-{fallback_count}"))
    seconds = time.perf_counter() - started
    return leaves[:max_width], compute_tree_stats(leaves[:max_width], fallback_count, pruned_count, seconds)


def make_tree_rollout_func(config: Any):
    def rollout_func(prompts: list[Any], trainer: Any) -> dict[str, Any]:
        tokenizer = trainer.processing_class
        rng = random.Random(config.seed + int(getattr(trainer.state, "global_step", 0)))
        all_prompt_ids: list[list[int] | None] = [None] * len(prompts)
        all_completion_ids: list[list[int] | None] = [None] * len(prompts)
        all_logprobs: list[list[float] | None] = [None] * len(prompts)
        all_tree_metadata: list[dict[str, Any] | None] = [None] * len(prompts)
        stats: list[TreeSamplingStats] = []

        grouped_indices: list[list[int]] = []
        group_by_key: dict[str, list[int]] = {}
        for idx, prompt in enumerate(prompts):
            key = _prompt_key(prompt)
            if key not in group_by_key:
                group_by_key[key] = []
                grouped_indices.append(group_by_key[key])
            group_by_key[key].append(idx)

        for group_index, prompt_indices in enumerate(grouped_indices):
            prompt = prompts[prompt_indices[0]]
            prompt_ids = _tokenize_prompt(prompt, tokenizer)

            def generate(input_ids: list[int]) -> tuple[list[int], list[float]]:
                remaining = max(1, config.max_completion_length - max(0, len(input_ids) - len(prompt_ids)))
                return _generate_segment_hf(
                    trainer,
                    input_ids,
                    max_new_tokens=min(config.tree_segment_length, remaining),
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                )

            leaves, tree_stats = sample_tree_for_prompt(
                prompt_ids=prompt_ids,
                tokenizer=tokenizer,
                generate_segment=generate,
                method=config.method,
                max_width=len(prompt_indices),
                max_depth=config.tree_max_depth,
                segment_length=config.tree_segment_length,
                fixed_step_width=config.tree_fixed_step_width,
                more_init_min=config.tree_more_init_min,
                more_init_max=config.tree_more_init_max,
                repetition_ngram_chars=config.tree_repetition_ngram_chars,
                repetition_count=config.tree_repetition_count,
                rng=rng,
            )
            stats.append(tree_stats)
            for rollout_index, (row_index, leaf) in enumerate(zip(prompt_indices, leaves, strict=False)):
                all_prompt_ids[row_index] = prompt_ids
                all_completion_ids[row_index] = leaf.completion_ids[: config.max_completion_length]
                all_logprobs[row_index] = leaf.logprobs[: config.max_completion_length]
                all_tree_metadata[row_index] = (
                    {
                        "prompt_index": group_index,
                        "row_index": row_index,
                        "rollout_index": rollout_index,
                        "tree_id": leaf.node_id,
                        "parent_id": leaf.parent_id,
                        "depth": leaf.depth,
                        "finish_reason": leaf.finish_reason,
                        "segments": [segment.__dict__ for segment in leaf.segments],
                    }
                )

        if stats:
            trainer._metrics["train"]["tree/active_nodes"].append(sum(s.active_nodes for s in stats) / len(stats))
            trainer._metrics["train"]["tree/fallback_count"].append(sum(s.fallback_count for s in stats) / len(stats))
            trainer._metrics["train"]["tree/pruned_count"].append(sum(s.pruned_count for s in stats) / len(stats))
            trainer._metrics["train"]["tree/mean_depth"].append(sum(s.mean_depth for s in stats) / len(stats))
            trainer._metrics["train"]["tree/rollout_seconds"].append(sum(s.rollout_seconds for s in stats))
            trainer._metrics["train"]["tree/tokenps"].append(sum(s.tokenps for s in stats) / len(stats))
            trainer._metrics["train"]["tree/trajps"].append(sum(s.trajps for s in stats) / len(stats))

        return {
            "prompt_ids": [ids or [] for ids in all_prompt_ids],
            "completion_ids": [ids or [] for ids in all_completion_ids],
            "logprobs": [logps or [] for logps in all_logprobs],
            "tree_metadata": [metadata or {} for metadata in all_tree_metadata],
        }

    return rollout_func
