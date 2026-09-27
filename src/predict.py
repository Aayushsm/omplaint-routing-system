"""
predict.py
----------
Loads the trained TF-IDF + Logistic Regression pipeline and exposes a
single function, `predict_category`, that performs STAGE 1 of the
system: turning raw complaint text into a predicted category + a
confidence score + the top-3 most likely categories.

This module does NOT know anything about departments or routing --
that is handled separately by routing.py (Stage 2). Keeping the two
stages in separate files/functions mirrors the two-stage architecture
described in the project report and makes it easy to explain in a viva.
"""

import os
import sys

import joblib
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import clean_text  # noqa: E402

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "model", "complaint_classifier.pkl")

_model_cache = {"pipeline": None}


class ModelNotFoundError(Exception):
    """Raised when the trained model file cannot be found on disk."""
    pass


def load_model():
    """
    Load the trained pipeline from disk (cached after first load so
    Streamlit doesn't reload it on every rerun).
    """
    if _model_cache["pipeline"] is not None:
        return _model_cache["pipeline"]

    if not os.path.exists(MODEL_PATH):
        raise ModelNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'. "
            f"Please run 'python src/train_model.py' first to train and save the model."
        )

    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        raise ModelNotFoundError(f"Failed to load the trained model: {e}")

    _model_cache["pipeline"] = pipeline
    return pipeline


def predict_category(raw_text: str, top_k: int = 3):
    """
    Predict the complaint category for a piece of raw complaint text.

    Parameters
    ----------
    raw_text : str
        The raw, unprocessed complaint text typed by the user.
    top_k : int
        How many top predictions (category + probability) to return.

    Returns
    -------
    dict with keys:
        "category"       : str  -- the single best predicted category
        "confidence"     : float -- probability of the best category (0-1)
        "top_predictions": list[(category, probability)] sorted descending
        "cleaned_text"    : str  -- the text after preprocessing (for display)

    Raises
    ------
    ValueError
        If raw_text is empty or becomes empty after cleaning.
    ModelNotFoundError
        If the trained model file is missing or fails to load.
    """
    if raw_text is None or not str(raw_text).strip():
        raise ValueError("Complaint text is empty. Please enter a complaint before submitting.")

    cleaned = clean_text(raw_text)
    if not cleaned:
        raise ValueError(
            "Complaint text became empty after cleaning (it may contain only "
            "symbols/numbers). Please enter a descriptive complaint."
        )

    pipeline = load_model()

    # predict_proba gives a probability for every class the model was
    # trained on -- this is what lets us report a genuine confidence
    # score and a top-3 breakdown instead of a single hard label.
    probabilities = pipeline.predict_proba([cleaned])[0]
    classes = pipeline.classes_

    order = np.argsort(probabilities)[::-1]  # descending order
    top_predictions = [(classes[i], float(probabilities[i])) for i in order[:top_k]]

    best_category, best_confidence = top_predictions[0]

    return {
        "category": best_category,
        "confidence": best_confidence,
        "top_predictions": top_predictions,
        "cleaned_text": cleaned,
    }
