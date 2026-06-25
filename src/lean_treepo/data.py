from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset

from .formatting import fits_prompt_length, has_formal_statement, preprocess_example


@dataclass(frozen=True)
class SplitSizes:
    train: int
    validation: int
    test: int


def deterministic_indices(total: int, sizes: SplitSizes, seed: int) -> dict[str, list[int]]:
    required = sizes.train + sizes.validation + sizes.test
    if required > total:
        raise ValueError(f"Requested {required} examples but dataset contains only {total}.")
    import random

    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_end = sizes.train
    val_end = train_end + sizes.validation
    return {
        "train": indices[:train_end],
        "validation": indices[train_end:val_end],
        "test": indices[val_end : val_end + sizes.test],
    }


def prepare_dataset(
    dataset_name: str,
    split: str,
    sizes: SplitSizes,
    seed: int,
    tokenizer: Any | None = None,
    max_prompt_length: int = 0,
) -> tuple[Dataset, Dataset | None, Dataset | None]:
    dataset = load_dataset(dataset_name, split=split, token=os.getenv("HF_TOKEN"))
    dataset = dataset.filter(has_formal_statement)
    remove_columns = list(dataset.column_names)
    dataset = dataset.map(preprocess_example, remove_columns=remove_columns)

    if tokenizer is not None and max_prompt_length > 0:
        dataset = dataset.filter(lambda row: fits_prompt_length(row, tokenizer, max_prompt_length))

    indices = deterministic_indices(len(dataset), sizes, seed)
    train = dataset.select(indices["train"])
    validation = dataset.select(indices["validation"]) if sizes.validation else None
    test = dataset.select(indices["test"]) if sizes.test else None
    return train, validation, test
