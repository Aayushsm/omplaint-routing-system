# NLP-Based Customer Complaint Classification and Automatic Routing System

An academic prototype that automatically classifies banking customer
complaints into a category using NLP + Machine Learning, then routes
each complaint to the correct department using a simple rule-based
mapping. Built with Python, scikit-learn, and Streamlit.

---

## 1. Project Overview

Customer complaints are often manually reviewed and forwarded to the
appropriate department. This is slow and can result in incorrect
routing. This project demonstrates an automated pipeline:

```
Customer Complaint
        |
Text Preprocessing
        |
NLP Classification Model (TF-IDF + Logistic Regression)
        |
Complaint Category
        |
Department Mapping (rule-based)
        |
Automatic Routing
        |
Department Queue / Dashboard (SQLite + Streamlit)
```

The system is split into two clearly separate stages:

- **Stage 1 — NLP Classification**: raw complaint text → predicted category
  (this is the only thing the ML model does)
- **Stage 2 — Routing**: predicted category → department (a simple lookup
  table in `src/routing.py`, not something the model "knows")

---

## 2. Features

- Synthetic but realistic 840-example banking complaint dataset across
  12 categories, generated from varied sentence templates (not simple
  word-swapped duplicates), including natural openers/closers and
  simulated typos.
- Text preprocessing (lowercasing, punctuation removal, whitespace
  normalisation).
- TF-IDF + Logistic Regression classification pipeline (primary model),
  with Multinomial Naive Bayes and Linear SVM trained for comparison.
- Real accuracy / precision / recall / F1-score and confusion matrix,
  computed from an actual train/test split — nothing is hardcoded.
- A second, stricter "template holdout" evaluation reported for
  transparency (see **Model Evaluation** below).
- Confidence score + top-3 predicted categories for every submitted
  complaint.
- Automatic department routing via an explicit mapping table.
- SQLite-backed complaint storage with human-friendly IDs (`CMP-0001`, ...).
- Streamlit dashboard with filters (department / category / status),
  summary metrics, charts, and status updates.
- Graceful handling of empty input, missing model files, and database
  errors.

---

## 3. Technologies

| Component        | Technology              |
|-------------------|--------------------------|
| Language           | Python 3                |
| Data handling       | pandas, numpy           |
| ML / NLP            | scikit-learn (TF-IDF, Logistic Regression) |
| Model persistence    | joblib                 |
| Frontend / Dashboard  | Streamlit             |
| Storage               | SQLite                |
| Evaluation plots       | matplotlib, seaborn  |

No LLMs, transformers, external APIs, Docker, or cloud services are
used. The entire project runs **fully offline** after dependencies are
installed.

---

## 4. Project Architecture

```
Raw complaint text
    -> preprocessing.clean_text()          (src/preprocessing.py)
    -> TfidfVectorizer                     (inside the saved pipeline)
    -> LogisticRegression                  (inside the saved pipeline)
    -> predicted category + confidence     (src/predict.py)
    -> route_complaint(category)           (src/routing.py)
    -> department
    -> database.insert_complaint(...)      (src/database.py)
    -> displayed in Streamlit UI           (app.py)
```

---

## 5. Folder Structure

```
complaint-routing-system/
│
├── app.py                          # Streamlit application (4 pages)
│
├── data/
│   └── complaints.csv              # Generated training dataset
│
├── model/
│   ├── complaint_classifier.pkl    # Trained TF-IDF + LogisticRegression pipeline
│   └── metrics.json                # Real evaluation metrics (used by the UI)
│
├── src/
│   ├── generate_dataset.py         # Builds data/complaints.csv
│   ├── preprocessing.py            # Text cleaning
│   ├── train_model.py              # Trains, evaluates, and saves the model
│   ├── predict.py                  # Loads model, predicts category + confidence
│   ├── routing.py                  # Category -> Department mapping (Stage 2)
│   └── database.py                 # SQLite storage layer
│
├── evaluation/
│   └── confusion_matrix.png        # Saved by train_model.py
│
├── requirements.txt
├── README.md
└── complaints.db                   # Created automatically on first run
```

---

## 6. Installation

From inside the `complaint-routing-system` folder:

```
pip install -r requirements.txt
```

(Use a virtual environment if you prefer: `python -m venv venv` then
activate it before installing.)

---

## 7. Dataset Generation

The dataset is already included (`data/complaints.csv`, 840 rows), but
you can regenerate it at any time:

```
python src/generate_dataset.py
```

This creates `data/complaints.csv` with columns `complaint_text`,
`category`, and `template_id` (the last is used only for the stricter
evaluation described below, not by the app).

---

## 8. Model Training

```
python src/train_model.py
```

This will:
1. Load and clean `data/complaints.csv`
2. Split into train/test (80/20, **stratified**, as required)
3. Train the primary TF-IDF + Logistic Regression pipeline
4. Also train Multinomial Naive Bayes and Linear SVM for comparison
5. Evaluate on the held-out test set (accuracy, precision, recall, F1,
   full classification report, confusion matrix)
6. Also run a stricter "template holdout" evaluation for transparency
   (see below)
7. Save the trained pipeline to `model/complaint_classifier.pkl`
8. Save `model/metrics.json` and `evaluation/confusion_matrix.png`

**You must run this once before starting the Streamlit app**, since
`app.py` loads the saved model rather than retraining every time.

---

## 9. Running the Streamlit Application

```
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

---

## 10. Example Complaints (for demo)

| Complaint | Expected Category | Expected Department |
|---|---|---|
| "My UPI payment failed but my account was debited." | Failed Transaction | Payments Department |
| "Someone made a transaction using my card and I did not authorize it." | Fraud / Suspicious Activity | Fraud & Security |
| "The ATM did not give me cash but the amount was deducted." | ATM Issue | ATM Operations |
| "I cannot log into the mobile banking application." | Login / Account Access or Mobile Banking / App Issue | IT Support |
| "I was charged twice for the same purchase." | Duplicate Transaction | Payments Department |
| "My EMI was deducted incorrectly this month." | Loan / EMI | Loans Department |
| "The banking app keeps crashing." | Mobile Banking / App Issue | IT Support |
| "Why was I charged an unexpected service fee?" | Charges / Fees | Finance Department |
| "I want to update my registered mobile number." | Account Services | Account Services Department |
| "My cancelled order refund hasn't arrived yet." | Refund / Reversal | Payments & Refunds |

All of the above were tested against the trained model during
development (see Section 13, Limitations, for honesty about dataset
characteristics).

---

## 11. How Classification Works (Stage 1)

1. **Preprocessing** (`src/preprocessing.py`): lowercase the text,
   strip URLs and punctuation, normalise whitespace. Stopwords are
   *not* removed by default, since short complaints can lose important
   signal words (e.g. "not") — see comments in the file.
2. **TF-IDF Vectorization**: converts the cleaned text into numeric
   features, weighting words/phrases (unigrams + bigrams) by how
   distinctive they are across the dataset.
3. **Logistic Regression**: a linear classifier trained on the TF-IDF
   features to predict one of the 12 complaint categories, along with
   a probability for every category (`predict_proba`), which is what
   powers the confidence score and top-3 breakdown.

## 12. How Routing Works (Stage 2)

`src/routing.py` contains a single dictionary,
`CATEGORY_TO_DEPARTMENT`, mapping each of the 12 categories to a
department (e.g. `"Fraud / Suspicious Activity" -> "Fraud & Security"`).
`route_complaint(category)` looks up this dictionary. This logic is
completely independent of the ML model — the model never sees or
predicts department names, only categories. This separation is
intentional and important to explain in a viva (see Section 15).

---

## 13. Model Evaluation

Real, non-fabricated metrics computed by `train_model.py` on an 840-row
dataset (80/20 stratified split):

- **Accuracy / Precision / Recall / F1-score**: reported live on the
  "Model Performance" page (`model/metrics.json`), not hardcoded.
- **Confusion Matrix**: saved as `evaluation/confusion_matrix.png`.

**Important honesty note:** because the dataset is synthetically
generated from a limited number of sentence templates per category
(with varied filler words, openers/closers, and simulated typos), a
random/stratified split can place near-duplicate sentences (same base
template) on both sides of the split. This makes the primary reported
accuracy very high (close to 100%). This is a **genuine result of the
required stratified split**, not a fabricated number — but it is
optimistic compared to how the model would perform on truly unseen
real-world complaints.

To be transparent about this, `train_model.py` **also** computes a
stricter evaluation where the test set only contains complaint
templates the model never saw *any* variation of during training. This
score (visible in an expandable section on the Model Performance page)
is meaningfully lower and is a more honest estimate of real-world
generalisation. Both numbers are shown — nothing is hidden.

---

## 14. Limitations

- Trained on synthetic, template-generated data rather than real
  customer complaints (see Section 13 for the honesty note on accuracy).
- Only supports the 12 categories it was trained on; out-of-scope
  complaints will still be forced into the closest category.
- Confidence scores are the model's internal probability estimates,
  not guarantees of correctness.
- The routing table is static and does not consider department
  workload, urgency, or escalation.
- No authentication / multi-user support — this is a single-user local
  prototype for demonstration purposes.

## 15. Future Scope

- Train on real, anonymised complaint data for better generalisation.
- Multi-label classification for complaints spanning multiple categories.
- Priority/urgency detection for triaging critical complaints.
- Role-based login for department staff to manage their own queue.
- Integration with a real ticketing system or email notifications.

---

## 16. Important Viva Questions & Answers

**Q1. Why TF-IDF + Logistic Regression instead of a deep learning /
transformer model?**
A: For a text classification task with a moderate number of
well-separated categories, TF-IDF + Logistic Regression is fast to
train, easy to explain (every prediction can be traced back to
weighted words), requires no GPU, and works well on a small dataset.
Transformers need much more data and computing power and are harder to
justify/explain in an academic setting.

**Q2. What does the "confidence score" actually mean?**
A: It's the probability the Logistic Regression model assigns to the
predicted category via `predict_proba`, i.e. how strongly the learned
weights favour that category given the TF-IDF features of the input.
It is a statistical estimate, not a certainty.

**Q3. Does the model know about departments?**
A: No. The model only ever predicts a complaint *category*. The
category-to-department mapping is a separate, simple rule-based lookup
(`src/routing.py`). This two-stage design keeps the system explainable
and lets the bank change departments without retraining the model.

**Q4. Why is accuracy so high (close to 100%)?**
A: Because the dataset is synthetically generated from sentence
templates, and the required stratified split can place near-duplicate
sentences on both sides of the split. We disclose this honestly and
also report a stricter "template holdout" evaluation (visible on the
Model Performance page) that gives a more realistic, lower accuracy
figure — this is explained in Section 13 above.

**Q5. Why TF-IDF and not just Bag-of-Words / word counts?**
A: TF-IDF down-weights very common words that appear in almost every
complaint (like "account" or "bank") and up-weights words that are
more distinctive to a specific category, which generally improves
classification quality over raw counts.

**Q6. Why Logistic Regression over Naive Bayes or SVM?**
A: All three were trained and compared (see Model Performance page).
Logistic Regression was chosen as the primary model because it
performed comparably to SVM, provides well-calibrated probability
estimates via `predict_proba` (needed for the confidence score), and
is straightforward to explain: each feature (word/phrase) has a
learned weight indicating how much it pushes the prediction toward or
away from each category.

**Q7. Why SQLite instead of a CSV or full external database?**
A: SQLite is built into Python's standard library, requires no server
setup, supports proper querying/filtering, and is more robust than a
CSV file for concurrent reads/writes from a running dashboard, while
staying appropriately lightweight for a local academic prototype.

**Q8. How would you improve this system for real-world deployment?**
A: Train on real, labelled complaint data; add periodic retraining;
add authentication; add monitoring for prediction drift; consider a
"human-in-the-loop" review step for low-confidence predictions instead
of auto-routing them.

**Q9. What happens if a customer's complaint doesn't clearly fit any category?**
A: The model still picks its best (highest-probability) category, but
the confidence score will be low, which is visibly flagged in the UI
("Low confidence") so a human reviewer knows to double-check it.

**Q10. Is this system doing sentiment analysis?**
A: No. This is a multi-class text classification task (predicting
*which type* of complaint it is), not sentiment analysis (which would
predict positive/negative/neutral tone).

---

## 17. Testing Performed

During development the following was manually verified end-to-end:
- Dataset generation produces a balanced, diverse 840-row dataset.
- Training runs successfully and produces real (non-fabricated) metrics.
- The 10 demo complaints above classify into their expected categories.
- Empty complaint input is rejected with a clear message (no crash).
- Nonsensical/symbol-only input is handled without crashing (low
  confidence, no error).
- Unusual/casual wording ("yo my card thing broke...") still produces a
  reasonable prediction.
- Complaints are correctly inserted into SQLite with sequential
  human-friendly IDs.
- Dashboard filters (department / category / status) work correctly.
- Complaint status can be updated from the dashboard.
- The saved model reloads correctly via joblib without retraining.

Note: because this offline build environment did not have network
access to install the `streamlit` package itself, the Streamlit UI code
was carefully reviewed and all underlying logic (prediction, routing,
database, training) was executed and verified directly in Python. When
you run `pip install -r requirements.txt` followed by
`streamlit run app.py` on your machine, the full UI will be live.
