
import os
import pandas as pd
from typing import Optional
from .utils import convert_columns_to_list


def merge_datasets(
    base_dataset_path: str,
    additional_dataset_csv: str,
    output_path: str,
    shuffle: bool = True,
    remove_exclamation_question: bool = True
) -> None:

    base_df = pd.read_pickle(base_dataset_path)
    additional_df = pd.read_csv(additional_dataset_csv)

    additional_df = convert_columns_to_list(additional_df)

    if remove_exclamation_question:
        additional_df = additional_df[
            additional_df["output"].apply(
                lambda x: not (
                    (4 in x) or  # !
                    (3 in x)     # ؟
                )
            )
        ]

    final_df = pd.concat(
        [base_df[["input", "output"]],
         additional_df[["input", "output"]]],
        ignore_index=True
    )

    if shuffle:
        final_df = final_df.sample(
            frac=1,
            random_state=42
        ).reset_index(drop=True)

    final_df.to_pickle(output_path)



def rebalance_with_rare_classes(
    base_dataset_path: str,
    rare_dataset_path: str,
    output_path: str,
    base_sample_size: int = 150_000
) -> None:

    base_df = pd.read_pickle(base_dataset_path)
    rare_df = pd.read_pickle(rare_dataset_path)

    excl_label = 4
    ques_label = 3

    data_excl = base_df[
        base_df["output"].apply(lambda x: excl_label in x)
    ]

    data_ques = base_df[
        base_df["output"].apply(
            lambda x: (ques_label in x) and (excl_label not in x)
        )
    ]

    rest = base_df[
        ~(base_df.index.isin(data_excl.index) |
          base_df.index.isin(data_ques.index))
    ]

    rest_sampled = rest.sample(
        n=min(base_sample_size, len(rest)),
        random_state=42
    )

    combined = pd.concat(
        [data_excl, rest_sampled, data_ques, rare_df],
        ignore_index=True
    )

    combined = combined.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    combined.to_pickle(output_path)