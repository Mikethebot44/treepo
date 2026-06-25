from collections import defaultdict

from lean_treepo.tree_sampler import has_repetition, initial_divergence, make_tree_rollout_func, sample_tree_for_prompt


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(char) % 20 + 1 for char in str(text)]}

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


def test_rollout_func_does_not_expand_repeated_prompt_batch(monkeypatch) -> None:
    from lean_treepo.config import config_from_args

    monkeypatch.setenv("KIMINA_BASE_URL", "http://kimina")
    config = config_from_args([
        "--method",
        "treepo_fixed_init",
        "--num-generations",
        "4",
        "--tree-segment-length",
        "1",
        "--tree-max-depth",
        "1",
    ])

    def fake_generate_segment_hf(*_args, **_kwargs):
        return [1], [-0.1]

    monkeypatch.setattr("lean_treepo.tree_sampler._generate_segment_hf", fake_generate_segment_hf)

    class State:
        global_step = 0

    class Trainer:
        processing_class = FakeTokenizer()
        state = State()
        _metrics = {"train": defaultdict(list)}

    prompts = ["same prompt"] * 4
    result = make_tree_rollout_func(config)(prompts, Trainer())
    assert len(result["completion_ids"]) == len(prompts)
    assert len(result["prompt_ids"]) == len(prompts)
    assert all(len(ids) == 1 for ids in result["completion_ids"])
