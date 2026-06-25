from __future__ import annotations

from typing import Any

from .tree_advantage import tree_token_advantages


class TreePOGRPOTrainerMixin:
    """Small extension that replaces sequence advantages with segment-level tree advantages."""

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        output = super()._generate_and_score_completions(inputs)
        if not getattr(self, "use_tree_advantage", False):
            return output

        metadata = getattr(self, "_last_tree_metadata", None)
        reward_func = getattr(self, "tree_reward", None)
        if reward_func is None and getattr(self, "reward_funcs", None):
            reward_func = self.reward_funcs[0]
        traces = getattr(reward_func, "last_traces", None)
        if not metadata or not traces:
            return output

        rewards = [float(trace.reward) for trace in traces]
        lengths = output["completion_mask"].sum(dim=1).to("cpu").tolist()
        lengths = [int(length) for length in lengths]
        if len(rewards) != len(lengths) or len(metadata) != len(lengths):
            return output

        shaped = tree_token_advantages(
            rewards=rewards,
            tree_metadata=metadata,
            completion_lengths=lengths,
            num_generations=self.num_generations,
        )

        import torch

        padded = torch.zeros_like(output["completion_mask"], dtype=torch.float32)
        for row_idx, row in enumerate(shaped):
            if not row:
                continue
            values = torch.tensor(row[: padded.shape[1]], dtype=torch.float32, device=padded.device)
            padded[row_idx, : values.numel()] = values
        output["advantages"] = padded
        return output


def build_treepo_trainer_class():
    from trl import GRPOTrainer

    class TreePOGRPOTrainer(TreePOGRPOTrainerMixin, GRPOTrainer):
        pass

    return TreePOGRPOTrainer
