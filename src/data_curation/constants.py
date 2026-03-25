# app/data_curation/constants.py

LABELS = {
    "O": 0,
    ".": 1,
    "،": 2,
    "؟": 3,
    "!": 4,
    "؛": 5,
    ":": 6
}

ID2LABEL = {v: k for k, v in LABELS.items()}

COMMA = LABELS["،"]
SEMICOLON = LABELS["؛"]
COLON = LABELS[":"]
EXCLAMATION = LABELS["!"]
QUESTION = LABELS["؟"]