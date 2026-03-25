
import argparse
import pandas as pd

from src.data_curation.utils import build_key_column
from src.data_curation.complex_sampler import extract_complex_samples
from src.data_curation.dataset_merger import merge_datasets


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dataset", required=True)
    parser.add_argument("--full_dataset_csv", required=True)
    parser.add_argument("--temp_complex_csv", required=True)
    parser.add_argument("--final_output", required=True)

    args = parser.parse_args()

    base_df = pd.read_pickle(args.base_dataset)
    base_df = build_key_column(base_df)

    used_keys = set(base_df["key"]) 

    collected = extract_complex_samples(
        full_dataset_csv=args.full_dataset_csv,
        used_keys=used_keys,
        output_csv=args.temp_complex_csv
    )

    print(f"Collected {collected} complex samples.")

    merge_datasets(
        base_dataset_path=args.base_dataset,
        additional_dataset_csv=args.temp_complex_csv,
        output_path=args.final_output
    )

    print("Final dataset created successfully.")


if __name__ == "__main__":
    main()