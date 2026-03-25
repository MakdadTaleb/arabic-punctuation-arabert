# Arabic Punctuation Restoration with AraBERT

[![Model on HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/makdadTaleb/arabic-punctuation-arabert)

Fine-tuning AraBERT for automatic punctuation restoration in Arabic text — a sequence labeling task that predicts the punctuation mark following each word.

---

## Overview

Arabic text is often written or transmitted without punctuation, making it harder to read and process automatically. This project builds a model that restores punctuation marks given unpunctuated Arabic input.

The task is framed as **token classification**: for each word in a sentence, predict one of 7 labels:

| Label | Symbol | Description |
|-------|--------|-------------|
| O | — | No punctuation |
| 1 | `.` | Period |
| 2 | `،` | Arabic comma |
| 3 | `؟` | Question mark |
| 4 | `!` | Exclamation mark |
| 5 | `؛` | Arabic semicolon |
| 6 | `:` | Colon |

---

## Pretrained Model
 
The fine-tuned model is available on Hugging Face:
 
🤗 **[makdadTaleb/arabic-punctuation-arabert](https://huggingface.co/makdadTaleb/arabic-punctuation-arabert)**
 
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
 
tokenizer = AutoTokenizer.from_pretrained("makdadTaleb/arabic-punctuation-arabert")
model = AutoModelForTokenClassification.from_pretrained("makdadTaleb/arabic-punctuation-arabert")
```
 
---

## Dataset

**Source:** [SSAC-UNPC](https://github.com/amirAlobeid/SSAC-UNPC) — a large-scale Arabic corpus of UN proceedings.

The raw corpus contains long documents with inconsistent punctuation formatting. A full preprocessing pipeline was built from scratch:

1. **Normalization** — unified Unicode variants (e.g. `?` → `؟`, `,` → `،`)
2. **Cleaning** — removed decorative quotes, normalized whitespace
3. **Sentence splitting** — rule-based regex with Arabic abbreviation handling
4. **Tokenization** — separated punctuation as standalone tokens
5. **Label creation** — each word gets the label of the punctuation that follows it

### Handling Class Imbalance

The raw corpus is heavily dominated by periods and commas. To address this:

- Extracted all sentences containing `!` and `؟` from the full corpus (rare classes)
- Extracted sentences rich in `،` and `؛` combinations
- Progressively grew the dataset across experiments:

| Version | Size |
|---------|------|
| Initial | 200k sentences |
| + rare punctuation | 200k → augmented |
| + comma/semicolon-rich | 300k sentences |
| Final | **400k sentences** |

---

## Model

**Base model:** [`aubmindlab/bert-base-arabertv02`](https://huggingface.co/aubmindlab/bert-base-arabertv02)

**Architecture:** AraBERT + linear classification head (7 classes)

**Training details:**
- Optimizer: AdamW (lr = 2e-5)
- Scheduler: Linear warmup (10%) + linear decay
- Loss: Weighted CrossEntropyLoss (to further address class imbalance)
- Mixed precision: AMP (FP16)
- Early stopping: patience = 2
- Epochs: 3
- Batch size: 16
- Max sequence length: 128

### Two-Stage Decision for Arabic Comma

The model showed systematic confusion between `،` (comma) and `؛` (semicolon). A custom post-processing step was added: if the model's confidence for `،` exceeds a threshold τ = 0.70 and is higher than `O`, it predicts comma directly; otherwise, it selects the highest-confidence non-comma label. This improved comma F1 from 0.685 → **0.749**.

---

## Experiments

Four experiments were conducted, varying dataset size and class weights:

| Experiment | Dataset | Weighted Avg F1 | Comma F1 |
|------------|---------|-----------------|----------|
| Exp 1 | 200k | 0.960 | 0.430 |
| Exp 2 | 200k (adjusted weights) | 0.944 | 0.528 |
| Exp 3 | 300k | 0.950 | 0.592 |
| Exp 4 | 400k | 0.950 | 0.685 |
| **Exp 4 + Two-Stage** | **400k** | **0.961** | **0.749** |

---

## Final Results (Exp 4 + Two-Stage)

Evaluated on a held-out validation set (~1.2M tokens, 10% of full dataset):

```
              precision    recall  f1-score   support

           O      0.995     0.964     0.979  1081801
           .      0.993     0.999     0.996    37183
           ،      0.646     0.891     0.749    76522
           ؟      0.955     0.965     0.960     5599
           !      0.520     0.361     0.426       72
           ؛      0.432     0.792     0.559     3960
           :      0.705     0.919     0.798     3485

    accuracy                          0.960  1208622
   macro avg      0.750     0.841     0.781  1208622
weighted avg      0.970     0.960     0.963  1208622
```

---

## Project Structure

arabic-punctuation-restoration/
│
├── api/
│ └── app.py # FastAPI application (inference API)
│
├── notebooks/
│ └── data_punctuation_analysiz.py # Data exploration and analysis (EDA)
│
├── scripts/
│ ├── build_dataset.py # Build base dataset
│ ├── build_complex_dataset.py # Extract and merge complex samples
│ ├── build_rare_dataset.py # Extract rare punctuation samples
│ ├── train_arabert.py # Train AraBERT model
│ ├── evaluate_model.py # Evaluate trained model
│ └── predict.py # Run inference on new text
│
├── src/
│ ├── data_curation/
│ │ ├── complex_sampler.py # Extract complex punctuation samples
│ │ ├── dataset_merger.py # Merge datasets
│ │ ├── utils.py # Helper functions (keys, formatting)
│ │ └── constants.py # Punctuation label constants
│ │
│ ├── preprocessing/
│ │ ├── cleaner.py # Text cleaning
│ │ ├── tokenizer.py # Tokenization logic
│ │ ├── sentence_splitter.py # Sentence segmentation
│ │ └── label_encoder.py # Encode punctuation labels
│ │
│ ├── models/
│ │ ├── arabert_model.py # Load and configure AraBERT
│ │ └── dataset.py # PyTorch Dataset class
│ │
│ ├── training/
│ │ ├── trainer.py # Training loop
│ │ ├── losses.py # Loss functions
│ │ ├── metrics.py # Evaluation metrics
│ │ └── early_stopping.py # Early stopping logic
│ │
│ ├── inference/
│ │ ├── predictor.py # Inference pipeline
│ │ └── postprocessing.py # Output formatting
│ │
│ └── evaluation/
│ ├── metrics_report.py # Metrics reporting
│ └── confusion.py # Confusion matrix
│
├── config.yaml # Project configuration
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## References

- AraBERT: [aubmindlab/bert-base-arabertv02](https://huggingface.co/aubmindlab/bert-base-arabertv02)
- Dataset: SSAC-UNPC Arabic Corpus

