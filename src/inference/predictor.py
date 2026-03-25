import torch
from transformers import AutoTokenizer
from src.models.arabert_model import load_model
from .postprocessing import two_stage_decision
import yaml
import os


class PunctuationPredictor:

    def __init__(self, model_path: str, config_path: str = "config.yaml", device=None):
        
        # ✅ قراءة الإعدادات من config
        self.config = self._load_config(config_path)
        self.model_name = self.config["model"]["name"]
        self.num_labels = self.config["data"]["num_labels"]
        self.use_two_stage = self.config["inference"].get("use_two_stage", True)
        self.comma_threshold = self.config["inference"].get("comma_threshold", 0.6)
        
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # ✅ تحميل التوكنيزر بالاسم الديناميكي
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # ✅ تحميل الموديل
        self.model = load_model(self.num_labels, self.model_name)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        # ✅ خريطة التسميات (ثابتة حسب constants.py)
        # LABELS = {"O":0, ".":1, "،":2, "؟":3, "!":4, "؛":5, ":":6}
        self.id2label = {
            0: "",      # NO_PUNCT (O)
            1: ".",     # PERIOD (.)
            2: "،",     # COMMA (،)
            3: "؟",     # QUESTION (؟)
            4: "!",     # EXCLAMATION (!)
            5: "؛",     # SEMICOLON (؛)
            6: ":"      # COLON (:)
        }

    def _load_config(self, path: str) -> dict:
        """تحميل ملف الإعدادات"""
        # البحث عن الملف في عدة مواقع
        possible_paths = [
            path,
            "config.yaml",
            "../config.yaml",
            "../../config.yaml",
            os.path.join(os.path.dirname(__file__), "../../config.yaml")
        ]
        
        for p in possible_paths:
            if os.path.exists(p):
                with open(p, "r") as f:
                    return yaml.safe_load(f)
        
        raise FileNotFoundError(f"Config file not found. Searched in: {possible_paths}")

    def predict(self, text: str, use_two_stage: bool = None) -> str:
        """
        التنبؤ بالترقيم
        
        Args:
            text: النص العربي
            use_two_stage: استخدام التحسين الثنائي (None = استخدام الإعداد من config)
        """
        if use_two_stage is None:
            use_two_stage = self.use_two_stage

        words = text.split()

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits

            if use_two_stage:
                preds = two_stage_decision(logits, comma_index=2, o_index=0, threshold=self.comma_threshold)
            else:
                preds = torch.argmax(logits, dim=-1)

        word_ids = encoding.word_ids(batch_index=0)

        final_output = []
        previous_word_idx = None

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue

            if word_idx != previous_word_idx:
                word = words[word_idx]
                label_id = preds[0][idx].item()
                punct = self.id2label[label_id]

                final_output.append(word + punct)

            previous_word_idx = word_idx

        return " ".join(final_output)