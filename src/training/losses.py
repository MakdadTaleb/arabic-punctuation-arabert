import torch


def get_weighted_loss(class_weights, device):
    weights = torch.tensor(class_weights, device=device)
    return torch.nn.CrossEntropyLoss(ignore_index=-100, weight=weights)