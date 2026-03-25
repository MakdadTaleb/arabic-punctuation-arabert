
import ast
from collections import Counter
import pandas as pd


def safe_literal_eval(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x


def convert_columns_to_list(df: pd.DataFrame) -> pd.DataFrame:
    df["input"] = df["input"].apply(safe_literal_eval)
    df["output"] = df["output"].apply(safe_literal_eval)
    return df


def build_key_column(df: pd.DataFrame) -> pd.DataFrame:
    df["key"] = df["input"].apply(lambda x: " ".join(x))
    return df


def count_label_distribution(df: pd.DataFrame) -> Counter:
    counter = Counter()
    for seq in df["output"]:
        for lbl in seq:
            if lbl != 0:
                counter[lbl] += 1
    return counter