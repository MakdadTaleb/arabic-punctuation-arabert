import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

DEFAULT_MODEL_NAME = "aubmindlab/bert-base-arabertv02"


def load_tokenizer(model_name: str = None):
    """تحميل التوكنيزر"""
    name = model_name or DEFAULT_MODEL_NAME
    return AutoTokenizer.from_pretrained(name)


def load_model(num_labels: int, model_name: str = None):
    """تحميل الموديل"""
    name = model_name or DEFAULT_MODEL_NAME
    model = AutoModelForTokenClassification.from_pretrained(
        name,
        num_labels=num_labels
    )
    return model