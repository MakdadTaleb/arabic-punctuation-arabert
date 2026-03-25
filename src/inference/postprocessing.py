import torch
import torch.nn.functional as F


def two_stage_decision(logits, comma_index=2, o_index=0, threshold=0.70):
    """
    Apply two-stage decision specifically for the Arabic comma class (،).

    Original logic from training experiments:
    - If comma probability > threshold AND comma > O probability → predict comma
    - Otherwise → pick the best non-comma label

    This is stronger than simple thresholding because when the model is
    uncertain about a comma, it selects the best alternative instead of
    defaulting blindly to O.

    Args:
        logits: model output (B, T, C)
        comma_index: label index for Arabic comma (،) — default 2
        o_index: label index for no punctuation (O) — default 0
        threshold: confidence threshold for comma prediction — default 0.70

    Returns:
        preds: (B, T) predicted label indices
    """
    probs = F.softmax(logits, dim=-1)

    preds = []
    for token_probs in probs.view(-1, probs.size(-1)):
        p_comma = token_probs[comma_index]
        p_o = token_probs[o_index]

        if p_comma > threshold and p_comma > p_o:
            preds.append(comma_index)
        else:
            # Remove comma from consideration, pick best remaining label
            token_probs_copy = token_probs.clone()
            token_probs_copy[comma_index] = -1.0
            preds.append(token_probs_copy.argmax().item())

    preds = torch.tensor(preds, device=logits.device)
    return preds.view(logits.size(0), logits.size(1))