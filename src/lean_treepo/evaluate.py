from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .data import SplitSizes, prepare_dataset
from .formatting import extract_lean_from_completion
from .lean_client import KiminaLeanClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Lean TreePO adapter with Kimina verification.")
    parser.add_argument("--model-or-adapter", required=True)
    parser.add_argument("--base-model", default=os.getenv("MODEL_NAME_OR_PATH", "Pythagoras-LM/Pythagoras-Prover-4B"))
    parser.add_argument("--dataset-name", default=os.getenv("DATASET_NAME", "AI-MO/NuminaMath-LEAN"))
    parser.add_argument("--dataset-split", default=os.getenv("DATASET_SPLIT", "train"))
    parser.add_argument("--test-size", type=int, default=int(os.getenv("EVAL_MAX_SAMPLES", "128")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("EVAL_N_SAMPLES", "16")))
    parser.add_argument("--max-prompt-length", type=int, default=int(os.getenv("MAX_PROMPT_LENGTH", "1024")))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("MAX_COMPLETION_LENGTH", "7168")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "1.0")))
    parser.add_argument("--top-p", type=float, default=float(os.getenv("TOP_P", "1.0")))
    parser.add_argument("--kimina-base-url", default=os.getenv("KIMINA_BASE_URL"))
    parser.add_argument("--kimina-api-key", default=os.getenv("KIMINA_API_KEY"))
    parser.add_argument("--output-path", default=os.getenv("EVAL_OUTPUT_PATH", "/workspace/outputs/treepo-runs/eval.jsonl"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.kimina_base_url:
        raise ValueError("KIMINA_BASE_URL is required.")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=os.getenv("HF_TOKEN"), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=os.getenv("HF_TOKEN"),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.model_or_adapter, token=os.getenv("HF_TOKEN"))
    model.eval()

    _, _, test_dataset = prepare_dataset(
        args.dataset_name,
        args.dataset_split,
        SplitSizes(train=1, validation=0, test=args.test_size),
        args.seed,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
    )
    if test_dataset is None:
        raise ValueError("No test dataset was produced.")

    client = KiminaLeanClient(args.kimina_base_url, args.kimina_api_key)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    correct_any = 0
    correct_majority = 0

    for example in test_dataset:
        prompt_ids = tokenizer.apply_chat_template(example["prompt"], tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        completions = []
        rewards = []
        for sample_idx in range(args.n_samples):
            with torch.no_grad():
                generated = model.generate(
                    prompt_ids,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completion_ids = generated[0, prompt_ids.shape[1] :].tolist()
            text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            extracted = extract_lean_from_completion(text, formal_statement=example["formal_statement"])
            verified = False
            feedback = "No Lean code extracted."
            if extracted.ok:
                result = client.verify(extracted.code, custom_id=f"{example.get('uuid', 'row')}-{sample_idx}")
                verified = result.success
                feedback = result.feedback
            completions.append({"text": text, "extraction_mode": extracted.mode, "verified": verified, "feedback": feedback})
            rewards.append(1 if verified else 0)
        any_correct = int(any(rewards))
        maj_correct = int(sum(rewards) > len(rewards) / 2)
        correct_any += any_correct
        correct_majority += maj_correct
        rows.append({"uuid": example.get("uuid", ""), "pass_at_k": any_correct, "majority_at_k": maj_correct, "completions": completions})

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = max(1, len(rows))
    print(f"pass@{args.n_samples}={correct_any / total:.4f}")
    print(f"majority@{args.n_samples}={correct_majority / total:.4f}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
