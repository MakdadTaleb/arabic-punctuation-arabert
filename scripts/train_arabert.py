import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.model.dataset import PunctuationDataset
from src.model.arabert_model import load_model, load_tokenizer
from src.training.losses import get_weighted_loss
from src.training.trainer import Trainer
from src.training.early_stopping import EarlyStopping

import yaml
import os



def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def main(config_path):

    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_PATH = config["data"]["path"]
    BATCH_SIZE = config["data"]["batch_size"]
    NUM_LABELS = config["data"]["num_labels"]
    VAL_SPLIT = config["data"]["val_split"]

    EPOCHS = config["training"]["epochs"]
    LEARNING_RATE = config["training"]["learning_rate"]
    WARMUP_RATIO = config["training"]["warmup_ratio"]
    PATIENCE = config["training"]["early_stopping_patience"]
    
    MODEL_SAVE_PATH = config["model"]["save_path"]
    MODEL_NAME = config["model"]["name"]

    class_weights = config["training"]["class_weights"]

    
    # =====================
    # Load dataset
    # =====================
    df = pd.read_pickle(DATA_PATH)

    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SPLIT,
        random_state=42,
        shuffle=True
    )

    # =====================
    # Model & Tokenizer
    # =====================
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, NUM_LABELS).to(device)

    # =====================
    # Datasets & Loaders
    # =====================
    train_dataset = PunctuationDataset(train_df, tokenizer)
    val_dataset = PunctuationDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # =====================
    # Optimizer & Scheduler
    # =====================
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps
    )

    # =====================
    # Loss
    # =====================
    criterion = get_weighted_loss(class_weights, device)

    # =====================
    # Trainer
    # =====================
    trainer = Trainer(model, optimizer, scheduler, criterion, device)
    early_stopping = EarlyStopping(patience=PATIENCE)
    history = []
    best_val_loss = float("inf")

    # =====================
    # Training Loop
    # =====================
    for epoch in range(EPOCHS):

        train_loss, train_acc = trainer.train_one_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        print(
            f"\nEpoch {epoch+1}/{EPOCHS}\n"
            f"Train → loss: {train_loss:.4f} | acc: {train_acc:.4f}\n"
            f"Val   → loss: {val_loss:.4f} | acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        if early_stopping.step(val_loss):
            print("Early stopping triggered.")
            break


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )

    args = parser.parse_args()

    main(args.config)