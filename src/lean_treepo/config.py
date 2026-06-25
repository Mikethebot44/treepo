from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

METHODS = ("grpo", "grpo_tree_sampling", "treepo_fixed_init", "treepo_more_init")
TREE_METHODS = ("grpo_tree_sampling", "treepo_fixed_init", "treepo_more_init")
TREE_ADVANTAGE_METHODS = ("treepo_fixed_init", "treepo_more_init")


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    return default if raw in (None, "") else int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw in (None, "") else float(raw)


def env_str(name: str, default: str | None) -> str | None:
    raw = os.getenv(name)
    return default if raw in (None, "") else raw


def comma_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def method_suffix(method: str) -> str:
    return method.replace("_", "-")


@dataclass(frozen=True)
class TrainConfig:
    method: str
    model_name_or_path: str
    dataset_name: str
    dataset_split: str
    output_root: str
    output_dir: str
    hub_model_prefix: str | None
    hub_model_id: str | None
    push_to_hub: bool
    hub_private_repo: bool
    hf_token_required: bool
    train_size: int
    val_size: int
    test_size: int
    seed: int
    max_steps: int
    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    warmup_steps: int | None
    logging_steps: int
    save_steps: int
    eval_steps: int
    save_total_limit: int
    max_prompt_length: int
    max_completion_length: int
    num_generations: int
    generation_batch_size: int | None
    temperature: float
    top_p: float
    top_k: int
    beta: float
    loss_type: str
    scale_rewards: str
    bf16: bool
    use_4bit: bool
    use_peft: bool
    gradient_checkpointing: bool
    attn_implementation: str
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    report_to: str
    wandb_project: str
    run_name: str
    log_completions: bool
    log_completions_steps: int
    kimina_base_url: str | None
    kimina_api_key: str | None
    kimina_timeout_seconds: float
    kimina_max_workers: int
    tree_segment_length: int
    tree_max_depth: int
    tree_fixed_step_width: int
    tree_more_init_min: int
    tree_more_init_max: int
    tree_repetition_ngram_chars: int
    tree_repetition_count: int

    @property
    def uses_tree_sampling(self) -> bool:
        return self.method in TREE_METHODS

    @property
    def uses_tree_advantage(self) -> bool:
        return self.method in TREE_ADVANTAGE_METHODS

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"METHOD must be one of {', '.join(METHODS)}.")
        if self.train_size <= 0:
            raise ValueError("TRAIN_SIZE must be positive.")
        if self.val_size < 0 or self.test_size < 0:
            raise ValueError("VAL_SIZE and TEST_SIZE must be non-negative.")
        if self.num_generations < 2:
            raise ValueError("NUM_GENERATIONS must be at least 2 for GRPO.")
        if self.max_steps == 0:
            raise ValueError("MAX_STEPS=0 is invalid; use a positive value or -1 for epoch training.")
        if self.push_to_hub and not self.hub_model_id:
            raise ValueError("PUSH_TO_HUB=true requires HUB_MODEL_ID or HUB_MODEL_PREFIX.")
        if (self.push_to_hub or self.hf_token_required) and not os.getenv("HF_TOKEN"):
            raise ValueError("HF_TOKEN is required for this training job.")
        if not self.kimina_base_url:
            raise ValueError("KIMINA_BASE_URL is required for verifier rewards.")
        if self.uses_tree_sampling:
            if self.tree_segment_length <= 0:
                raise ValueError("TREE_SEGMENT_LENGTH must be positive for TreePO methods.")
            if self.tree_max_depth <= 0:
                raise ValueError("TREE_MAX_DEPTH must be positive for TreePO methods.")
            if self.tree_fixed_step_width <= 0:
                raise ValueError("TREE_FIXED_STEP_WIDTH must be positive for TreePO methods.")
        if self.method == "treepo_more_init" and self.tree_more_init_min > self.tree_more_init_max:
            raise ValueError("TREE_MORE_INIT_MIN must be <= TREE_MORE_INIT_MAX.")


def _derive_output_dir(output_root: str, method: str) -> str:
    cleaned_root = output_root.rstrip("/\\")
    return f"{cleaned_root}/{method}"


def _derive_hub_model_id(prefix: str | None, method: str) -> str | None:
    if not prefix:
        return None
    return f"{prefix}-{method_suffix(method)}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Pythagoras Prover 4B with GRPO/TreePO on NuminaMath-LEAN.")
    parser.add_argument("--method", choices=METHODS, default=env_str("METHOD", "grpo"))
    parser.add_argument("--model-name-or-path", default=env_str("MODEL_NAME_OR_PATH", "Pythagoras-LM/Pythagoras-Prover-4B"))
    parser.add_argument("--dataset-name", default=env_str("DATASET_NAME", "AI-MO/NuminaMath-LEAN"))
    parser.add_argument("--dataset-split", default=env_str("DATASET_SPLIT", "train"))
    parser.add_argument("--output-root", default=env_str("OUTPUT_ROOT", "/workspace/outputs/treepo-runs"))
    parser.add_argument("--output-dir", default=env_str("OUTPUT_DIR", None))
    parser.add_argument("--hub-model-prefix", default=env_str("HUB_MODEL_PREFIX", None))
    parser.add_argument("--hub-model-id", default=env_str("HUB_MODEL_ID", None))
    parser.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=env_bool("PUSH_TO_HUB", False))
    parser.add_argument("--hub-private-repo", action=argparse.BooleanOptionalAction, default=env_bool("HUB_PRIVATE_REPO", True))
    parser.add_argument("--hf-token-required", action=argparse.BooleanOptionalAction, default=env_bool("HF_TOKEN_REQUIRED", False))
    parser.add_argument("--train-size", type=int, default=env_int("TRAIN_SIZE", 64))
    parser.add_argument("--val-size", type=int, default=env_int("VAL_SIZE", 16))
    parser.add_argument("--test-size", type=int, default=env_int("TEST_SIZE", 16))
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))
    parser.add_argument("--max-steps", type=int, default=env_int("MAX_STEPS", 10))
    parser.add_argument("--num-train-epochs", type=float, default=env_float("NUM_TRAIN_EPOCHS", 1.0))
    parser.add_argument("--per-device-train-batch-size", type=int, default=env_int("PER_DEVICE_TRAIN_BATCH_SIZE", 1))
    parser.add_argument("--per-device-eval-batch-size", type=int, default=env_int("PER_DEVICE_EVAL_BATCH_SIZE", 1))
    parser.add_argument("--gradient-accumulation-steps", type=int, default=env_int("GRADIENT_ACCUMULATION_STEPS", 16))
    parser.add_argument("--learning-rate", type=float, default=env_float("LEARNING_RATE", 1e-6))
    parser.add_argument("--warmup-ratio", type=float, default=env_float("WARMUP_RATIO", 0.03))
    parser.add_argument("--warmup-steps", type=int, default=env_int("WARMUP_STEPS", None))
    parser.add_argument("--logging-steps", type=int, default=env_int("LOGGING_STEPS", 1))
    parser.add_argument("--save-steps", type=int, default=env_int("SAVE_STEPS", 10))
    parser.add_argument("--eval-steps", type=int, default=env_int("EVAL_STEPS", 10))
    parser.add_argument("--save-total-limit", type=int, default=env_int("SAVE_TOTAL_LIMIT", 3))
    parser.add_argument("--max-prompt-length", type=int, default=env_int("MAX_PROMPT_LENGTH", 1024))
    parser.add_argument("--max-completion-length", type=int, default=env_int("MAX_COMPLETION_LENGTH", 7168))
    parser.add_argument("--num-generations", type=int, default=env_int("NUM_GENERATIONS", 16))
    parser.add_argument("--generation-batch-size", type=int, default=env_int("GENERATION_BATCH_SIZE", None))
    parser.add_argument("--temperature", type=float, default=env_float("TEMPERATURE", 1.0))
    parser.add_argument("--top-p", type=float, default=env_float("TOP_P", 1.0))
    parser.add_argument("--top-k", type=int, default=env_int("TOP_K", 0))
    parser.add_argument("--beta", type=float, default=env_float("BETA", 0.0))
    parser.add_argument("--loss-type", default=env_str("LOSS_TYPE", "dapo"))
    parser.add_argument("--scale-rewards", default=env_str("SCALE_REWARDS", "group"))
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=env_bool("BF16", True))
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=env_bool("USE_4BIT", True))
    parser.add_argument("--use-peft", action=argparse.BooleanOptionalAction, default=env_bool("USE_PEFT", True))
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=env_bool("GRADIENT_CHECKPOINTING", True))
    parser.add_argument("--attn-implementation", default=env_str("ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--bnb-4bit-quant-type", default=env_str("BNB_4BIT_QUANT_TYPE", "nf4"))
    parser.add_argument("--bnb-4bit-compute-dtype", default=env_str("BNB_4BIT_COMPUTE_DTYPE", "bfloat16"))
    parser.add_argument("--lora-r", type=int, default=env_int("LORA_R", 64))
    parser.add_argument("--lora-alpha", type=int, default=env_int("LORA_ALPHA", 128))
    parser.add_argument("--lora-dropout", type=float, default=env_float("LORA_DROPOUT", 0.05))
    parser.add_argument("--lora-target-modules", default=env_str("LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"))
    parser.add_argument("--report-to", default=env_str("REPORT_TO", "none"))
    parser.add_argument("--wandb-project", default=env_str("WANDB_PROJECT", "lean-treepo"))
    parser.add_argument("--run-name", default=env_str("RUN_NAME", None))
    parser.add_argument("--log-completions", action=argparse.BooleanOptionalAction, default=env_bool("LOG_COMPLETIONS", False))
    parser.add_argument("--log-completions-steps", type=int, default=env_int("LOG_COMPLETIONS_STEPS", 100))
    parser.add_argument("--kimina-base-url", default=env_str("KIMINA_BASE_URL", None))
    parser.add_argument("--kimina-api-key", default=env_str("KIMINA_API_KEY", None))
    parser.add_argument("--kimina-timeout-seconds", type=float, default=env_float("KIMINA_TIMEOUT_SECONDS", 60.0))
    parser.add_argument("--kimina-max-workers", type=int, default=env_int("KIMINA_MAX_WORKERS", 8))
    parser.add_argument("--tree-segment-length", type=int, default=env_int("TREE_SEGMENT_LENGTH", 512))
    parser.add_argument("--tree-max-depth", type=int, default=env_int("TREE_MAX_DEPTH", 14))
    parser.add_argument("--tree-fixed-step-width", type=int, default=env_int("TREE_FIXED_STEP_WIDTH", 2))
    parser.add_argument("--tree-more-init-min", type=int, default=env_int("TREE_MORE_INIT_MIN", 2))
    parser.add_argument("--tree-more-init-max", type=int, default=env_int("TREE_MORE_INIT_MAX", 8))
    parser.add_argument("--tree-repetition-ngram-chars", type=int, default=env_int("TREE_REPETITION_NGRAM_CHARS", 32))
    parser.add_argument("--tree-repetition-count", type=int, default=env_int("TREE_REPETITION_COUNT", 8))
    return parser


def config_from_args(argv: list[str] | None = None) -> TrainConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_root = args.output_root
    output_dir = args.output_dir or _derive_output_dir(output_root, args.method)
    hub_model_id = args.hub_model_id or _derive_hub_model_id(args.hub_model_prefix, args.method)
    run_name = args.run_name or f"pythagoras-prover-4b-numina-lean-{method_suffix(args.method)}"
    config = TrainConfig(
        method=args.method,
        model_name_or_path=args.model_name_or_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        output_root=output_root,
        output_dir=output_dir,
        hub_model_prefix=args.hub_model_prefix,
        hub_model_id=hub_model_id,
        push_to_hub=args.push_to_hub,
        hub_private_repo=args.hub_private_repo,
        hf_token_required=args.hf_token_required,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        beta=args.beta,
        loss_type=args.loss_type,
        scale_rewards=args.scale_rewards,
        bf16=args.bf16,
        use_4bit=args.use_4bit,
        use_peft=args.use_peft,
        gradient_checkpointing=args.gradient_checkpointing,
        attn_implementation=args.attn_implementation,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=comma_list(args.lora_target_modules),
        report_to=args.report_to,
        wandb_project=args.wandb_project,
        run_name=run_name,
        log_completions=args.log_completions,
        log_completions_steps=args.log_completions_steps,
        kimina_base_url=args.kimina_base_url,
        kimina_api_key=args.kimina_api_key,
        kimina_timeout_seconds=args.kimina_timeout_seconds,
        kimina_max_workers=args.kimina_max_workers,
        tree_segment_length=args.tree_segment_length,
        tree_max_depth=args.tree_max_depth,
        tree_fixed_step_width=args.tree_fixed_step_width,
        tree_more_init_min=args.tree_more_init_min,
        tree_more_init_max=args.tree_more_init_max,
        tree_repetition_ngram_chars=args.tree_repetition_ngram_chars,
        tree_repetition_count=args.tree_repetition_count,
    )
    config.validate()
    return config
