from lean_treepo.config import TrainConfig, config_from_args


def make_config(**overrides):
    data = dict(
        method="grpo",
        model_name_or_path="Pythagoras-LM/Pythagoras-Prover-4B",
        dataset_name="AI-MO/NuminaMath-LEAN",
        dataset_split="train",
        output_root="/workspace/outputs/treepo-runs",
        output_dir="/workspace/outputs/treepo-runs/grpo",
        hub_model_prefix=None,
        hub_model_id=None,
        push_to_hub=False,
        hub_private_repo=True,
        hf_token_required=False,
        train_size=64,
        val_size=16,
        test_size=16,
        seed=42,
        max_steps=10,
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-6,
        warmup_ratio=0.03,
        warmup_steps=None,
        logging_steps=1,
        save_steps=10,
        eval_steps=10,
        save_total_limit=3,
        max_prompt_length=1024,
        max_completion_length=7168,
        num_generations=16,
        generation_batch_size=None,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        beta=0.0,
        loss_type="dapo",
        scale_rewards="group",
        bf16=True,
        use_4bit=True,
        use_peft=True,
        gradient_checkpointing=True,
        attn_implementation="sdpa",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        lora_target_modules=["q_proj"],
        report_to="none",
        wandb_project="lean-treepo",
        run_name="run",
        log_completions=False,
        log_completions_steps=100,
        kimina_base_url="http://kimina",
        kimina_api_key=None,
        kimina_timeout_seconds=60.0,
        kimina_max_workers=8,
        tree_segment_length=512,
        tree_max_depth=14,
        tree_fixed_step_width=2,
        tree_more_init_min=2,
        tree_more_init_max=8,
        tree_repetition_ngram_chars=32,
        tree_repetition_count=8,
    )
    data.update(overrides)
    return TrainConfig(**data)


def test_valid_config() -> None:
    make_config().validate()


def test_push_requires_hub_id() -> None:
    try:
        make_config(push_to_hub=True, hub_model_id=None).validate()
    except ValueError as exc:
        assert "HUB_MODEL_ID" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_tree_method_requires_positive_segment_length() -> None:
    try:
        make_config(method="treepo_fixed_init", tree_segment_length=0).validate()
    except ValueError as exc:
        assert "TREE_SEGMENT_LENGTH" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_blank_output_dir_derives_method_specific_path(monkeypatch) -> None:
    monkeypatch.setenv("KIMINA_BASE_URL", "http://kimina")
    monkeypatch.setenv("OUTPUT_ROOT", "/workspace/outputs/treepo-runs")
    monkeypatch.setenv("OUTPUT_DIR", "")
    config = config_from_args(["--method", "treepo_fixed_init", "--train-size", "1", "--val-size", "0", "--test-size", "0", "--num-generations", "2"])
    assert config.output_dir == "/workspace/outputs/treepo-runs/treepo_fixed_init"
