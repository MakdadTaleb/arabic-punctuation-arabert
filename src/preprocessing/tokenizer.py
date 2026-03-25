
import re
from .constants import PUNCTS_PATTERN


def tokenize(text: str) -> list[str]:
    text = re.sub(fr"([{PUNCTS_PATTERN}])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()