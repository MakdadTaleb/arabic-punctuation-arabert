
import os
import pandas as pd
from tqdm import tqdm
from typing import Set

from .constants import COMMA, SEMICOLON, COLON
from .utils import convert_columns_to_list, build_key_column


def is_complex_sequence(labels: list[int]) -> bool:

    has_complex_punct = (
        labels.count(COMMA) >= 2 or
        labels.count(SEMICOLON) >= 2 or
        (labels.count(COMMA) >= 1 and labels.count(SEMICOLON) >= 1)
    )

    has_complex_colon = (
        (labels.count(COLON) >= 1 and labels.count(COMMA) >= 1) or
        (labels.count(COLON) >= 1 and labels.count(SEMICOLON) >= 1)
    )

    return has_complex_punct or has_complex_colon


def extract_complex_samples(
    full_dataset_csv: str,
    used_keys: Set[str],
    output_csv: str,
    target_size: int = 300_000,
    chunk_size: int = 10_000,
    min_length: int = 20
) -> int:

    total_collected = 0

    for chunk in tqdm(
        pd.read_csv(full_dataset_csv, chunksize=chunk_size),
        desc="Extracting complex samples",
        unit="chunk"
    ):

        if total_collected >= target_size:
            break

        chunk = convert_columns_to_list(chunk)

        mask = chunk["output"].apply(
            lambda x: is_complex_sequence(x) and len(x) >= min_length
        )

        filtered = chunk[mask].copy()

        if filtered.empty:
            continue

        filtered = build_key_column(filtered)
        filtered = filtered[~filtered["key"].isin(used_keys)]

        if filtered.empty:
            continue

        remaining = target_size - total_collected
        filtered = filtered.iloc[:remaining]

        used_keys.update(filtered["key"])

        filtered = filtered.drop(columns=["key"])

        filtered[["input", "output"]].to_csv(
            output_csv,
            mode="a",
            header=not os.path.exists(output_csv),
            index=False
        )

        total_collected += len(filtered)

    return total_collected