
import os
import gc
import pandas as pd
from typing import Generator

from .text_cleaner import clean_text
from .sentence_splitter import split_sentences
from .tokenizer import tokenize
from .label_encoder import create_labels


def iter_text_files(folder_path: str) -> Generator[str, None, None]:
    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            yield os.path.join(folder_path, fname)


def process_single_file(file_path: str) -> pd.DataFrame:
    rows = []

    with open(file_path, "r", encoding="utf-8-sig") as f:
        raw = f.read()

    cleaned = clean_text(raw)
    sentences = split_sentences(cleaned)

    for sentence in sentences:
        tokens = tokenize(sentence)
        inp, out = create_labels(tokens)

        if not inp:
            continue

        has_punct = any(x != 0 for x in out)

        rows.append({
            "input": inp,
            "output": out,
            "input_len": len(inp),
            "has_punct": has_punct
        })

    return pd.DataFrame(rows)


def build_dataset(
    input_folder: str,
    output_csv_path: str,
    min_len: int = 5,
    max_len: int = 60
) -> None:

    for file_path in iter_text_files(input_folder):
        df_part = process_single_file(file_path)

        df_part = df_part[
            (df_part["input_len"] >= min_len) &
            (df_part["has_punct"] == True) &
            (df_part["input_len"] <= max_len)
        ]

        if len(df_part) > 0:
            df_part[["input", "output"]].to_csv(
                output_csv_path,
                mode="a",
                header=not os.path.exists(output_csv_path),
                index=False
            )

        del df_part
        gc.collect()