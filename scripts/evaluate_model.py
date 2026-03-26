import torch
import pandas as pd
import json
import numpy as np
import yaml
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.model.dataset import PunctuationDataset
from src.model.arabert_model import load_model, load_tokenizer
from src.evaluation.metrics_report import generate_classification_report
from src.evaluation.confusion import compute_confusion_matrix


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def flatten_predictions(preds, labels):
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    mask = labels != -100
    return preds[mask], labels[mask]


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            preds, labels = flatten_predictions(
                preds.cpu().numpy(),
                labels.cpu().numpy()
            )

            all_preds.extend(preds)
            all_labels.extend(labels)

    return np.array(all_preds), np.array(all_labels)


def main():
    config = load_config("config.yaml")
    
    DATA_PATH = config["data"]["path"]
    MODEL_PATH = config["model"]["save_path"]
    BATCH_SIZE = config["data"]["batch_size"]
    NUM_LABELS = config["data"]["num_labels"]
    MODEL_NAME = config["model"]["name"] 

    LABEL_NAMES = [
        "NO_PUNCT",    # 0
        "PERIOD",      # 1 (.)
        "COMMA",       # 2 (،)
        "SEMICOLON",   # 3 (؛)
        "COLON",       # 4 (:)
        "QUESTION",    # 5 (؟)
        "EXCLAMATION"  # 6 (!)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ تحميل البيانات
    df = pd.read_pickle(DATA_PATH)
    _, val_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)

    # ✅ تحميل الموديل والتوكنيزر بالاسم الديناميكي
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(NUM_LABELS, MODEL_NAME)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    val_dataset = PunctuationDataset(val_df, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # ✅ التقييم
    print("Running evaluation...")
    preds, labels = evaluate(model, val_loader, device)

    # ✅ النتائج
    report = generate_classification_report(labels, preds, LABEL_NAMES)
    cm = compute_confusion_matrix(labels, preds)
    accuracy = (preds == labels).mean()

    print("\n" + "="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print("="*50)
    print("\nClassification Report:\n")
    print(report)

    # ✅ حفظ النتائج
    np.save("confusion_matrix.npy", cm)
    
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    with open("metrics.json", "w") as f:
        json.dump({
            "accuracy": float(accuracy),
            "model_name": MODEL_NAME,
            "num_samples": len(val_df)
        }, f, indent=4)

    print("\n✅ Evaluation artifacts saved.")


if __name__ == "__main__":
    main()