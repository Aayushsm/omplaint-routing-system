"""
preprocessing.py
-----------------
Basic, explainable text preprocessing for customer complaints.

The preprocessing is deliberately kept LIGHT. Aggressive preprocessing
(e.g. heavy stemming, removing all "stopwords" including negations like
"not") can strip away words that matter for classification
(e.g. "not working" vs "working"). For a TF-IDF + Logistic Regression
pipeline, light cleaning generally works best.

Steps performed:
1. Lowercasing
2. Removing punctuation / special characters (keeping letters, numbers,
   and spaces)
3. Normalising whitespace
4. Optional stopword removal (OFF by default -- see note below)

Note on stopwords:
We intentionally do NOT remove stopwords by default because short
complaint sentences ("my card is not working") lose important signal
words ("not") when common stopword lists are applied. The
TfidfVectorizer's own `min_df`/`max_df` and n-gram settings already
reduce the influence of uninformative words, which is a gentler and
more transparent approach for a college-project viva.
"""

import re


def clean_text(text: str) -> str:
    """
    Clean a single raw complaint string.

    Parameters
    ----------
    text : str
        Raw complaint text as typed by the customer.

    Returns
    -------
    str
        Cleaned, lowercased text with normalised whitespace, ready to be
        fed into the TF-IDF vectorizer.
    """
    if text is None:
        return ""

    text = str(text)

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs (defensive, in case a complaint contains a link)
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # 3. Remove characters that are not letters, digits, or spaces.
    #    We keep digits because amounts / dates can carry meaning
    #    (e.g. "charged twice", "rs 500").
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 4. Normalise whitespace (collapse multiple spaces, strip ends)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_batch(texts):
    """Apply clean_text to a list/Series of complaint strings."""
    return [clean_text(t) for t in texts]
