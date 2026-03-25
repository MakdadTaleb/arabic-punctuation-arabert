import torch


def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    mask = labels != -100

    correct = (preds[mask] == labels[mask]).sum().item()
    total = mask.sum().item()

    return correct / total if total > 0 else 0.0