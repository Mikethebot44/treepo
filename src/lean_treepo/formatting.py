from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

REQUIRED_COLUMNS = ("formal_statement",)

LEAN_ONLY_INSTRUCTION = (
    "Complete the following Lean 4 theorem. Return a complete Lean 4 file that Kimina can compile. "
    "Do not use tools. Do not include markdown unless you use a single Lean code block."
)

LEAN_FENCE_RE = re.compile(r"```(?:lean4?|Lean4?)?\s*(.*?)```", re.DOTALL)
IMPORT_MATHLIB_RE = re.compile(r"(import\s+Mathlib\b[\s\S]*)", re.MULTILINE)
DECL_START_RE = re.compile(r"(?m)^\s*(?:theorem|lemma|example)\b[\s\S]*")
PROOF_BODY_HINT_RE = re.compile(r"(?m)^\s*(?:by|calc\b|have\b|intro\b|rintro\b|simp\b|aesop\b|omega\b|linarith\b|nlinarith\b)")


@dataclass(frozen=True)
class LeanExtraction:
    code: str
    mode: str
    ok: bool


def has_formal_statement(example: dict[str, Any]) -> bool:
    return isinstance(example.get("formal_statement"), str) and bool(example["formal_statement"].strip())


def normalize_lean(text: str) -> str:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    return "\n".join(lines).strip()


def make_prompt(example: dict[str, Any]) -> list[dict[str, str]]:
    statement = normalize_lean(str(example["formal_statement"]))
    return [
        {
            "role": "user",
            "content": f"{LEAN_ONLY_INSTRUCTION}\n\n```lean4\n{statement}\n```",
        }
    ]


def preprocess_example(example: dict[str, Any]) -> dict[str, Any]:
    if not has_formal_statement(example):
        raise ValueError("Example is missing non-empty formal_statement.")
    return {
        "prompt": make_prompt(example),
        "formal_statement": normalize_lean(str(example["formal_statement"])),
        "formal_ground_truth": normalize_lean(str(example.get("formal_ground_truth") or "")),
        "uuid": str(example.get("uuid") or ""),
        "source": str(example.get("source") or ""),
        "problem_type": str(example.get("problem_type") or ""),
        "exam": str(example.get("exam") or ""),
    }


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def extract_lean_from_completion(completion: Any, formal_statement: str | None = None) -> LeanExtraction:
    text = completion_to_text(completion).strip()
    if not text:
        return LeanExtraction("", "empty", False)

    fences = LEAN_FENCE_RE.findall(text)
    if fences:
        code = normalize_lean(fences[-1])
        return LeanExtraction(code, "lean_fence", bool(code))

    imports = list(IMPORT_MATHLIB_RE.finditer(text))
    if imports:
        code = normalize_lean(imports[-1].group(1))
        return LeanExtraction(code, "import_mathlib_block", bool(code))

    declarations = list(DECL_START_RE.finditer(text))
    if declarations:
        code = normalize_lean(declarations[-1].group(0))
        return LeanExtraction(code, "declaration_block", bool(code))

    if formal_statement:
        suffix = text
        if "```" in suffix:
            suffix = suffix.replace("```lean4", "").replace("```lean", "").replace("```", "")
        suffix = normalize_lean(suffix)
        if PROOF_BODY_HINT_RE.search(suffix):
            statement = normalize_lean(formal_statement)
            if statement.endswith(":= by") and suffix.startswith("by"):
                suffix = suffix[2:].strip()
            code = normalize_lean(f"{statement}\n  {suffix}")
            return LeanExtraction(code, "statement_plus_proof_suffix", True)

    return LeanExtraction("", "no_lean_found", False)


def tokenized_prompt_length(example: dict[str, Any], tokenizer: Any) -> int:
    prompt = example["prompt"]
    if hasattr(tokenizer, "apply_chat_template"):
        tokenized = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        if isinstance(tokenized, dict):
            tokenized = tokenized["input_ids"]
        return len(tokenized)
    text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in prompt)
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def fits_prompt_length(example: dict[str, Any], tokenizer: Any, max_prompt_length: int) -> bool:
    if max_prompt_length <= 0:
        return True
    return tokenized_prompt_length(example, tokenizer) <= max_prompt_length
