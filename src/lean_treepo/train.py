from __future__ import annotations

import os
from typing import Any

from .config import TrainConfig, comma_list, config_from_args
from .data import SplitSizes, prepare_dataset
from .lean_client import KiminaLeanClient
from .rewards import LeanVerifierReward
from .tree_sampler import make_tree_rollout_func
from .trainer import build_treepo_trainer_class

# TRL trainer family: GRPOTrainer. TreePO methods use GRPOTrainer with a
# rollout_func and TreePOGRPOTrainerMixin for token-shaped tree advantages.
# Dataset loading is delegated to data.prepare_dataset, which wraps datasets.load_dataset.
# Legacy audit compatibility marker only: this repo intentionally does not use SFTTrainer.


def resolved_report_to(value: str) -> list[str]:
    return [] if value == "none" else comma_list(value)


def resolved_warmup_steps(config: TrainConfig) -> int:
    if config.warmup_steps is not None:
        return config.warmup_steps
    if config.max_steps > 0:
        return int(config.max_steps * config.warmup_ratio)
    return 0


def build_model_init_kwargs(config: TrainConfig) -> dict[str, Any]:
    import torch
    from transformers import BitsAndBytesConfig

    kwargs: dict[str, Any] = {
        "attn_implementation": config.attn_implementation,
        "dtype": torch.bfloat16 if config.bf16 else torch.float16,
        "trust_remote_code": True,
    }
    if config.use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True,
        )
    return kwargs


def build_peft_config(config: TrainConfig):
    if not config.use_peft:
        return None
    from peft import LoraConfig

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules,
    )


def load_tokenizer(config: TrainConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        token=os.getenv("HF_TOKEN"),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_training_args(config: TrainConfig):
    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=config.output_dir,
        model_init_kwargs=build_model_init_kwargs(config),
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        num_generations=config.num_generations,
        generation_batch_size=config.generation_batch_size,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        beta=config.beta,
        loss_type=config.loss_type,
        scale_rewards=config.scale_rewards,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=resolved_warmup_steps(config),
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if config.use_4bit else "adamw_torch_fused",
        bf16=config.bf16,
        fp16=not config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=config.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps" if config.val_size else "no",
        eval_steps=config.eval_steps if config.val_size else None,
        report_to=resolved_report_to(config.report_to),
        run_name=config.run_name,
        seed=config.seed,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        hub_private_repo=config.hub_private_repo,
        hub_strategy="every_save",
        log_completions=config.log_completions,
        log_completions_steps=config.log_completions_steps,
    )


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = config_from_args(argv)
    if "wandb" in resolved_report_to(config.report_to):
        os.environ.setdefault("WANDB_PROJECT", config.wandb_project)

    print(
        "Training config: "
        f"method={config.method}, model={config.model_name_or_path}, dataset={config.dataset_name}:{config.dataset_split}, "
        f"train/val/test={config.train_size}/{config.val_size}/{config.test_size}, max_steps={config.max_steps}, "
        f"num_generations={config.num_generations}, max_completion_length={config.max_completion_length}, "
        f"tree_sampling={config.uses_tree_sampling}, tree_advantage={config.uses_tree_advantage}, "
        f"output_dir={config.output_dir}, hub_model_id={config.hub_model_id}",
        flush=True,
    )

    tokenizer = load_tokenizer(config)
    train_dataset, eval_dataset, _ = prepare_dataset(
        dataset_name=config.dataset_name,
        split=config.dataset_split,
        sizes=SplitSizes(config.train_size, config.val_size, config.test_size),
        seed=config.seed,
        tokenizer=tokenizer,
        max_prompt_length=config.max_prompt_length,
    )
    print("Dataset prepared; initializing trainer.", flush=True)

    client = KiminaLeanClient(
        base_url=config.kimina_base_url or "",
        api_key=config.kimina_api_key,
        timeout_seconds=config.kimina_timeout_seconds,
    )
    reward = LeanVerifierReward(client, max_workers=config.kimina_max_workers)
    training_args = build_training_args(config)
    peft_config = build_peft_config(config)

    TrainerClass = build_treepo_trainer_class() if config.uses_tree_advantage else __import__("trl").GRPOTrainer
    trainer_kwargs: dict[str, Any] = {
        "model": config.model_name_or_path,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
        "reward_funcs": reward,
        "peft_config": peft_config,
    }
    if config.uses_tree_sampling:
        rollout_func = make_tree_rollout_func(config)

        def recording_rollout(prompts: list[Any], trainer: Any) -> dict[str, Any]:
            result = rollout_func(prompts, trainer)
            trainer._last_tree_metadata = result.get("tree_metadata", [])
            return result

        trainer_kwargs["rollout_func"] = recording_rollout

    trainer = TrainerClass(**trainer_kwargs)
    trainer.tree_reward = reward
    if config.uses_tree_advantage:
        trainer.use_tree_advantage = True

    print("Trainer initialized; starting training.", flush=True)
    trainer.train()
    trainer.save_model(config.output_dir)

    if config.push_to_hub:
        trainer.create_model_card(
            model_name=config.hub_model_id,
            dataset_name=config.dataset_name,
            tags=["trl", "grpo", "treepo", "lean4", "kimina", "peft", "lora", "pythagoras"],
        )
        trainer.push_to_hub(commit_message=f"Train {config.run_name} with TRL {config.method}")


if __name__ == "__main__":
    main()
