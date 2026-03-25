
import argparse
from src.preprocessing.dataset_builder import build_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--min_len", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=60)

    args = parser.parse_args()

    build_dataset(
        input_folder=args.input_folder,
        output_csv_path=args.output_csv,
        min_len=args.min_len,
        max_len=args.max_len
    )


if __name__ == "__main__":
    main()