
import re
from .constants import NORMALIZE_MAP


def normalize_punctuation(text: str) -> str:
    for k, v in NORMALIZE_MAP.items():
        text = text.replace(k, v)
    return text


def clean_text(text: str) -> str:
    text = normalize_punctuation(text)
    text = re.sub(r'[«»“”]', '', text)
    text = re.sub(r'[^\S\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\t', ' ')
    return text.strip()