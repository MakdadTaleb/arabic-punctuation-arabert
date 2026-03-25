import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from .metrics import compute_accuracy


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.scaler = GradScaler()

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0

        for batch in tqdm(dataloader, desc="Training"):
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits

                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.scheduler.step()

            acc = compute_accuracy(logits, labels)

            total_loss += loss.item()
            total_acc += acc

        return total_loss / len(dataloader), total_acc / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits

                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )

                acc = compute_accuracy(logits, labels)

                total_loss += loss.item()
                total_acc += acc

        return total_loss / len(dataloader), total_acc / len(dataloader)