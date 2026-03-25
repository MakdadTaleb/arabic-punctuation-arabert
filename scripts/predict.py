import argparse
import os
from src.inference.predictor import PunctuationPredictor


def main():
    parser = argparse.ArgumentParser(description="Arabic Punctuation Restoration")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/best_arabert_weights_large.pt",
        help="Path to model weights"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f" Error: Model file not found: {args.model_path}")
        return
    
    if not os.path.exists(args.config):
        print(f" Error: Config file not found: {args.config}")
        return

    print(f"Loading model from: {args.model_path}")
    print(f"Using config: {args.config}")
    
    predictor = PunctuationPredictor(
        model_path=args.model_path,
        config_path=args.config
    )
    
    print(f"Model loaded: {predictor.model_name}")
    print("Ready for inference!\n")

    while True:
        text = input("\nEnter Arabic text (or 'exit'): ").strip()

        if text.lower() == "exit":
            print("Goodbye!")
            break
        
        if not text:
            print("Please enter some text.")
            continue

        try:
            output = predictor.predict(text)
            print("\nRestored Text:")
            print("-" * 50)
            print(output)
            print("-" * 50)
        except Exception as e:
            print(f" Error during prediction: {e}")


if __name__ == "__main__":
    main()