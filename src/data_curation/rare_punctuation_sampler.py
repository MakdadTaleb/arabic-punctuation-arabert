# app/data_curation/rare_punctuation_sampler.py

import os
import gc
import pandas as pd
from typing import Tuple

from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.sentence_splitter import split_sentences
from src.preprocessing.tokenizer import tokenize
from src.preprocessing.label_encoder import create_labels
from .constants import EXCLAMATION, QUESTION


def extract_rare_sentences_from_file(
    file_path: str
) -> Tuple[list, list, int, int]:

    collected_inputs = []
    collected_outputs = []

    excl_count = 0
    ques_count = 0

    with open(file_path, "r", encoding="utf-8-sig") as f:
        raw = f.read()

    cleaned = clean_text(raw)
    sentences = split_sentences(cleaned)

    for s in sentences:
        tokens = tokenize(s)
        inp, out = create_labels(tokens)

        if not inp:
            continue

        has_excl = EXCLAMATION in out
        has_ques = QUESTION in out

        if has_excl or has_ques:
            collected_inputs.append(inp)
            collected_outputs.append(out)

            excl_count += out.count(EXCLAMATION)
            ques_count += out.count(QUESTION)

    return collected_inputs, collected_outputs, excl_count, ques_count


def build_rare_punctuation_dataset(
    dataset_folder: str,
    output_path: str
) -> None:

    all_inputs = []
    all_outputs = []

    total_excl = 0
    total_ques = 0

    for filename in os.listdir(dataset_folder):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(dataset_folder, filename)

        inp, out, excl_c, ques_c = extract_rare_sentences_from_file(file_path)

        if inp:
            all_inputs.extend(inp)
            all_outputs.extend(out)

        total_excl += excl_c
        total_ques += ques_c

        gc.collect()

    df = pd.DataFrame({
        "input": all_inputs,
        "output": all_outputs
    })

    df.to_pickle(output_path)

    print(f"Total (!) detected: {total_excl}")
    print(f"Total (؟) detected: {total_ques}")