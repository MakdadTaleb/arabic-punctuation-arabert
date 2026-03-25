
import re

NORMALIZE_MAP = {
    "?": "؟",
    "؟": "؟",
    "!": "!",
    "¡": "!",
    ",": "،",
    "،": "،",
    ";": "؛",
    "؛": "؛",
    ":": ":",
    ".": "."
}

LABELS = {
    "O": 0,
    ".": 1,
    "،": 2,
    "؟": 3,
    "!": 4,
    "؛": 5,
    ":": 6
}

PUNCTS_PATTERN = "".join([re.escape(p) for p in NORMALIZE_MAP.values()])