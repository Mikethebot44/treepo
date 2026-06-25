from lean_treepo.formatting import extract_lean_from_completion, make_prompt, preprocess_example


def test_extracts_last_lean_fence() -> None:
    completion = "Reasoning...\n```lean\n#check Nat\n```\nFinal:\n```lean4\nimport Mathlib\n#check Int\n```"
    extracted = extract_lean_from_completion(completion)
    assert extracted.ok
    assert extracted.mode == "lean_fence"
    assert extracted.code == "import Mathlib\n#check Int"


def test_extracts_import_mathlib_block() -> None:
    completion = "The proof is below.\nimport Mathlib\ntheorem foo : True := by\n  trivial"
    extracted = extract_lean_from_completion(completion)
    assert extracted.ok
    assert extracted.mode == "import_mathlib_block"
    assert extracted.code.startswith("import Mathlib")


def test_extracts_declaration_block() -> None:
    completion = "We can prove it as follows.\ntheorem foo : True := by\n  trivial"
    extracted = extract_lean_from_completion(completion)
    assert extracted.ok
    assert extracted.mode == "declaration_block"
    assert extracted.code.startswith("theorem foo")


def test_appends_proof_suffix_to_statement() -> None:
    statement = "import Mathlib\ntheorem foo : True := by"
    completion = "by\n  trivial"
    extracted = extract_lean_from_completion(completion, formal_statement=statement)
    assert extracted.ok
    assert extracted.mode == "statement_plus_proof_suffix"
    assert extracted.code.endswith("trivial")


def test_no_code_returns_failed_extraction() -> None:
    extracted = extract_lean_from_completion("I do not know.")
    assert not extracted.ok
    assert extracted.mode == "no_lean_found"


def test_preprocess_prompt_uses_formal_statement_only() -> None:
    row = {"formal_statement": "import Mathlib\n#check Nat", "formal_ground_truth": "secret"}
    processed = preprocess_example(row)
    assert "secret" not in make_prompt(row)[0]["content"]
    assert processed["prompt"][0]["role"] == "user"
    assert "import Mathlib" in processed["prompt"][0]["content"]
