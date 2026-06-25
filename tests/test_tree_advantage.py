from lean_treepo.tree_advantage import group_grpo_advantages, tree_token_advantages


def test_group_grpo_advantages_zero_std_returns_zero() -> None:
    assert group_grpo_advantages([1.0, 1.0], num_generations=2) == [0.0, 0.0]


def test_tree_token_advantages_shape_and_sibling_assignment() -> None:
    metadata = [
        {"prompt_index": 0, "segments": [{"parent_id": "0", "depth": 1, "segment_start": 0, "segment_end": 2}]},
        {"prompt_index": 0, "segments": [{"parent_id": "0", "depth": 1, "segment_start": 0, "segment_end": 3}]},
    ]
    shaped = tree_token_advantages([0.0, 1.0], metadata, [2, 3], num_generations=2)
    assert len(shaped) == 2
    assert len(shaped[0]) == 2
    assert len(shaped[1]) == 3
    assert shaped[0][0] < 0
    assert shaped[1][0] > 0


def test_tree_advantage_falls_back_for_singleton_subgroup() -> None:
    metadata = [
        {"prompt_index": 0, "segments": [{"parent_id": "a", "depth": 1, "segment_start": 0, "segment_end": 1}]},
        {"prompt_index": 0, "segments": [{"parent_id": "b", "depth": 1, "segment_start": 0, "segment_end": 1}]},
    ]
    shaped = tree_token_advantages([0.0, 1.0], metadata, [1, 1], num_generations=2)
    assert shaped[0][0] < 0
    assert shaped[1][0] > 0
