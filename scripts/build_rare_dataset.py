# scripts/build_rare_dataset.py

import argparse
from src.data_curation.rare_punctuation_sampler import (
    build_rare_punctuation_dataset
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_folder", required=True)
    parser.add_argument("--output_path", required=True)

    args = parser.parse_args()

    build_rare_punctuation_dataset(
        dataset_folder=args.dataset_folder,
        output_path=args.output_path
    )


if __name__ == "__main__":
    main()