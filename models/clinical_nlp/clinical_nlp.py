"""
Clinical NLP Module for HealthRisk AI.

Implements clinical text processing using transformer-based models
for discharge note classification, named entity recognition, and
clinical complexity scoring.

When ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT) and PyTorch are
available, this module uses the full Transformer pipeline.  When they
are not installed, it falls back to a TF-IDF + Logistic Regression
baseline that still demonstrates the NLP architecture.

References:
- Alsentzer et al. (2019) Publicly Available Clinical BERT Embeddings
- Johnson et al. (2023) MIMIC-IV Clinical Database
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# Try importing deep learning libraries; fall back gracefully
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Clinical Text Preprocessing
# ---------------------------------------------------------------------------

# Common clinical abbreviation expansions
CLINICAL_ABBREVIATIONS = {
    "pt": "patient", "dx": "diagnosis", "hx": "history",
    "rx": "prescription", "tx": "treatment", "sx": "symptoms",
    "c/o": "complaining of", "s/p": "status post",
    "w/": "with", "w/o": "without", "b/l": "bilateral",
    "prn": "as needed", "bid": "twice daily", "tid": "three times daily",
    "qid": "four times daily", "qd": "daily", "po": "by mouth",
    "iv": "intravenous", "im": "intramuscular",
}


def preprocess_clinical_text(text: str) -> str:
    """
    Preprocess clinical note text for NLP analysis.

    - Lowercases text
    - Expands common clinical abbreviations
    - Removes excessive whitespace
    - Normalises numeric lab values

    Parameters
    ----------
    text : str
        Raw clinical note text.

    Returns
    -------
    str
        Preprocessed text ready for tokenisation.
    """
    text = text.lower().strip()

    # Expand abbreviations
    for abbrev, expansion in CLINICAL_ABBREVIATIONS.items():
        pattern = r'\b' + re.escape(abbrev) + r'\b'
        text = re.sub(pattern, expansion, text)

    # Normalise whitespace
    text = re.sub(r'\s+', ' ', text)

    # Mask specific numeric values to reduce vocabulary
    text = re.sub(r'\d+\.\d+', '<NUM>', text)
    text = re.sub(r'\b\d{3,}\b', '<NUM>', text)

    return text


# ---------------------------------------------------------------------------
# Named Entity Recognition (Rule-Based Baseline)
# ---------------------------------------------------------------------------

# Medication patterns
MEDICATION_PATTERNS = [
    r'\b(metformin|insulin|lisinopril|atorvastatin|amlodipine|metoprolol)\b',
    r'\b(omeprazole|levothyroxine|hydrochlorothiazide|losartan|gabapentin)\b',
    r'\b(furosemide|warfarin|aspirin|clopidogrel|glipizide|glyburide)\b',
    r'\b(pioglitazone|sitagliptin|empagliflozin|liraglutide|semaglutide)\b',
]

# Condition patterns (ICD-10 descriptions)
CONDITION_PATTERNS = [
    r'\b(diabetes|diabetic|dm|type 2 diabetes|t2dm)\b',
    r'\b(hypertension|htn|high blood pressure)\b',
    r'\b(heart failure|chf|hfref|hfpef)\b',
    r'\b(chronic kidney disease|ckd|renal failure)\b',
    r'\b(copd|chronic obstructive pulmonary)\b',
    r'\b(pneumonia|sepsis|uti|urinary tract infection)\b',
]

# Procedure patterns
PROCEDURE_PATTERNS = [
    r'\b(cardiac catheterization|cabg|pci|stent)\b',
    r'\b(dialysis|hemodialysis|peritoneal dialysis)\b',
    r'\b(intubation|ventilation|mechanical ventilation)\b',
    r'\b(ct scan|mri|x-ray|echocardiogram|echo)\b',
]


def extract_clinical_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract clinical named entities from text using pattern matching.

    This serves as a baseline NER system. In production, this would
    be replaced by a fine-tuned ClinicalBERT NER model.

    Parameters
    ----------
    text : str
        Clinical note text.

    Returns
    -------
    dict
        medications, conditions, procedures found in the text.
    """
    text_lower = text.lower()
    entities: Dict[str, List[str]] = {
        "medications": [],
        "conditions": [],
        "procedures": [],
    }

    for pattern in MEDICATION_PATTERNS:
        matches = re.findall(pattern, text_lower)
        entities["medications"].extend(matches)

    for pattern in CONDITION_PATTERNS:
        matches = re.findall(pattern, text_lower)
        entities["conditions"].extend(matches)

    for pattern in PROCEDURE_PATTERNS:
        matches = re.findall(pattern, text_lower)
        entities["procedures"].extend(matches)

    # De-duplicate
    for key in entities:
        entities[key] = list(set(entities[key]))

    return entities


# ---------------------------------------------------------------------------
# Clinical Text Classifier
# ---------------------------------------------------------------------------

class ClinicalTextClassifier:
    """
    Clinical text classifier for predicting discharge disposition
    or readmission risk from clinical notes.

    Uses ClinicalBERT when available, falls back to TF-IDF + LR.
    """

    def __init__(self, model_name: str = "tfidf_lr"):
        """
        Parameters
        ----------
        model_name : str
            Model to use: "clinicalbert" or "tfidf_lr".
        """
        self.model_name = model_name
        self.model = None
        self.classes_ = ["low_risk", "moderate_risk", "high_risk"]

        if model_name == "clinicalbert" and TRANSFORMERS_AVAILABLE:
            self._init_clinicalbert()
        else:
            self._init_tfidf_lr()

    def _init_clinicalbert(self):
        """Initialise ClinicalBERT model from HuggingFace."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT",
            num_labels=3,
        )

    def _init_tfidf_lr(self):
        """Initialise TF-IDF + Logistic Regression baseline."""
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words="english",
                min_df=2,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=1.0,
            )),
        ])
        self._is_fitted = False

    def fit(self, texts: List[str], labels: List[str]):
        """
        Train the classifier.

        Parameters
        ----------
        texts : list of str
            Clinical note texts.
        labels : list of str
            Classification labels.
        """
        processed = [preprocess_clinical_text(t) for t in texts]

        if self.model_name == "tfidf_lr" or not TRANSFORMERS_AVAILABLE:
            self.model.fit(processed, labels)
            self._is_fitted = True
            self.classes_ = list(self.model.classes_)

    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict class labels for clinical texts.

        Parameters
        ----------
        texts : list of str
            Clinical note texts.

        Returns
        -------
        list of str
            Predicted class labels.
        """
        processed = [preprocess_clinical_text(t) for t in texts]

        if self.model_name == "tfidf_lr" or not TRANSFORMERS_AVAILABLE:
            return list(self.model.predict(processed))

        # ClinicalBERT inference path
        return self._predict_clinicalbert(processed)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        texts : list of str
            Clinical note texts.

        Returns
        -------
        np.ndarray
            Shape (n_samples, n_classes) probability matrix.
        """
        processed = [preprocess_clinical_text(t) for t in texts]

        if self.model_name == "tfidf_lr" or not TRANSFORMERS_AVAILABLE:
            return self.model.predict_proba(processed)

        return self._predict_proba_clinicalbert(processed)

    def _predict_clinicalbert(self, texts: List[str]) -> List[str]:
        """ClinicalBERT inference."""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers not installed")

        inputs = self.tokenizer(
            texts, return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1)
        return [self.classes_[p] for p in preds.numpy()]

    def _predict_proba_clinicalbert(self, texts: List[str]) -> np.ndarray:
        """ClinicalBERT probability inference."""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers not installed")

        inputs = self.tokenizer(
            texts, return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        return probs.numpy()


# ---------------------------------------------------------------------------
# Clinical Complexity Scorer
# ---------------------------------------------------------------------------

def score_clinical_complexity(text: str) -> Dict[str, float]:
    """
    Score clinical complexity from note text.

    Uses entity density, medication count, and condition count
    as proxies for clinical complexity.

    Parameters
    ----------
    text : str
        Clinical note text.

    Returns
    -------
    dict
        complexity_score (0-1), entity_counts, word_count.
    """
    entities = extract_clinical_entities(text)

    med_count = len(entities["medications"])
    cond_count = len(entities["conditions"])
    proc_count = len(entities["procedures"])
    word_count = len(text.split())

    # Complexity heuristic (normalised 0-1)
    raw_score = (
        med_count * 0.15
        + cond_count * 0.25
        + proc_count * 0.20
        + min(word_count / 500, 1.0) * 0.40
    )
    complexity_score = min(1.0, raw_score)

    return {
        "complexity_score": round(complexity_score, 4),
        "medication_count": med_count,
        "condition_count": cond_count,
        "procedure_count": proc_count,
        "word_count": word_count,
        "entities": entities,
    }
