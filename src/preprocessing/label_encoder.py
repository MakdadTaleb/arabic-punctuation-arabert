
from .constants import LABELS


def create_labels(tokens: list[str]) -> tuple[list[str], list[int]]:
    input_tokens = []
    output_labels = []

    for i, tok in enumerate(tokens):
        if tok in LABELS:
            continue

        input_tokens.append(tok)
        next_tok = tokens[i + 1] if i + 1 < len(tokens) else ""

        if next_tok in LABELS:
            output_labels.append(LABELS[next_tok])
        else:
            output_labels.append(LABELS["O"])

    return input_tokens, output_labels