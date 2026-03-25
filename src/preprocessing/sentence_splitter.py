
import re


SENTENCE_SPLIT_PATTERN = r'''
    (?<!\b[دمنص]\.)
    (?<=[\.؟!?])
    ["»”)]*
    \s+
'''


def split_sentences(text: str) -> list[str]:
    sentences = re.split(
        SENTENCE_SPLIT_PATTERN,
        text,
        flags=re.VERBOSE
    )
    return [s.strip() for s in sentences if s.strip()]