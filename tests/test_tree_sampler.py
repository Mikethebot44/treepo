from lean_treepo.tree_sampler import has_repetition, initial_divergence, sample_tree_for_prompt


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(96 + token) for token in ids if token > 0)


def test_repetition_detector() -> None:
    assert has_repetition("abcd" * 10, ngram_chars=4, count_threshold=5)
    assert not has_repetition("abcdef", ngram_chars=4, count_threshold=5)


def test_initial_divergence_modes() -> None:
    import random

    rng = random.Random(1)
    assert initial_divergence("treepo_fixed_init", 2, 2, 8, rng) == 2
    value = initial_divergence("treepo_more_init", 2, 2, 8, rng)
    assert 2 <= value <= 8


def test_tree_sampler_respects_width_and_records_segments() -> None:
    calls = {"n": 0}

    def generate(_input_ids):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            return [0], [-0.1]
        return [1, 2], [-0.2, -0.3]

    import random

    leaves, stats = sample_tree_for_prompt(
        prompt_ids=[9],
        tokenizer=FakeTokenizer(),
        generate_segment=generate,
        method="treepo_fixed_init",
        max_width=4,
        max_depth=2,
        segment_length=2,
        fixed_step_width=2,
        more_init_min=2,
        more_init_max=8,
        repetition_ngram_chars=32,
        repetition_count=8,
        rng=random.Random(0),
    )
    assert len(leaves) == 4
    assert all(leaf.segments for leaf in leaves)
    assert stats.trajps >= 0
