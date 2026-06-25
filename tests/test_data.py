from lean_treepo.data import SplitSizes, deterministic_indices


def test_deterministic_indices_are_stable_and_disjoint() -> None:
    first = deterministic_indices(100, SplitSizes(10, 5, 3), seed=42)
    second = deterministic_indices(100, SplitSizes(10, 5, 3), seed=42)
    assert first == second
    all_indices = first["train"] + first["validation"] + first["test"]
    assert len(all_indices) == len(set(all_indices))
    assert len(first["train"]) == 10
    assert len(first["validation"]) == 5
    assert len(first["test"]) == 3


def test_deterministic_indices_reject_oversized_request() -> None:
    try:
        deterministic_indices(3, SplitSizes(2, 1, 1), seed=1)
    except ValueError as exc:
        assert "Requested" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
