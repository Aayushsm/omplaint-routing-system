"""
app.py
------
NLP-Based Customer Complaint Classification and Automatic Routing System
Main Streamlit application.

Run with:
    streamlit run app.py

Pages:
    1. Complaint Submission  -- enter a complaint, get classified + routed
    2. Complaint Dashboard   -- view / filter all stored complaints
    3. Model Performance     -- real accuracy/precision/recall/F1 + confusion matrix
    4. About Project         -- problem statement, objectives, tech stack
"""

import os
import sys
import json

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Make src/ importable regardless of working directory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from predict import predict_category, ModelNotFoundError  # noqa: E402
from routing import route_complaint, STATUS_OPTIONS, ALL_CATEGORIES  # noqa: E402
import database  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, "evaluation", "confusion_matrix.png")

st.set_page_config(
    page_title="Complaint Classification & Routing System",
    page_icon="🏦",
    layout="wide",
)

# Initialise the database (creates the table if it doesn't exist yet)
database.init_db()


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def confidence_badge(confidence: float) -> str:
    """Return a short qualitative label for a confidence score."""
    if confidence >= 0.75:
        return "High"
    elif confidence >= 0.45:
        return "Medium"
    else:
        return "Low"


# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.title("🏦 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Complaint Submission",
        "Complaint Dashboard",
        "Model Performance",
        "About Project",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "NLP-Based Customer Complaint Classification & Automatic Routing System"
)
st.sidebar.caption("Academic Prototype • Banking / Financial Domain")


# ==================================================================
# PAGE 1 -- COMPLAINT SUBMISSION
# ==================================================================
if page == "Complaint Submission":
    st.title("NLP-Based Customer Complaint Classification & Routing System")
    st.markdown(
        "Enter a customer complaint below. The system will classify it into a "
        "category using a trained NLP model, then automatically route it to the "
        "responsible department."
    )

    with st.form("complaint_form"):
        complaint_text = st.text_area(
            "Enter your customer complaint",
            placeholder="My UPI transaction failed but the amount was deducted from my account.",
            height=120,
        )
        submitted = st.form_submit_button("Classify & Route Complaint", type="primary")

    if submitted:
        try:
            result = predict_category(complaint_text)
            category = result["category"]
            confidence = result["confidence"]
            department = route_complaint(category)

            record = database.insert_complaint(
                complaint_text=complaint_text,
                predicted_category=category,
                confidence=confidence,
                department=department,
            )

            st.success("Complaint classified and routed successfully.")

            # ---- Result card ----
            col1, col2, col3 = st.columns(3)
            col1.metric("Complaint ID", record["complaint_id"])
            col2.metric("Predicted Category", category)
            col3.metric(
                "Confidence",
                f"{confidence * 100:.1f}%",
                help="Model's predicted probability for this category "
                     "(not a certainty -- see note below).",
            )

            col4, col5 = st.columns(2)
            col4.metric("Assigned Department", department)
            col5.metric("Status", record["status"])

            st.caption(f"Complaint ID: {record['complaint_id']}  |  Timestamp: {record['timestamp']}")

            conf_label = confidence_badge(confidence)
            st.info(
                f"**Model confidence level:** {conf_label} ({confidence * 100:.1f}%). "
                f"This is the model's estimated probability, not a guarantee of correctness."
            )

            # Top-3 predictions
            st.subheader("Top Predicted Categories")
            top_df = pd.DataFrame(
                result["top_predictions"], columns=["Category", "Probability"]
            )
            top_df["Probability"] = (top_df["Probability"] * 100).round(1).astype(str) + "%"
            st.table(top_df)

            # Explanation of the two-stage pipeline
            st.subheader("How this decision was made")
            st.markdown(
                f"1. **NLP Classification (Stage 1):** The complaint text was cleaned "
                f"and converted into TF-IDF features, and a Logistic Regression model "
                f"predicted the category **'{category}'** with **{confidence * 100:.1f}%** confidence.\n\n"
                f"2. **Routing (Stage 2):** The category **'{category}'** was looked up in the "
                f"category-to-department mapping table and routed to **'{department}'**.\n\n"
                f"*Note: the ML model only predicts the complaint category. It does not "
                f"understand departments or organisational structure -- that mapping is handled "
                f"by a separate, simple routing rule (see routing.py), which keeps the system "
                f"transparent and easy to modify.*"
            )

        except ValueError as ve:
            st.warning(str(ve))
        except ModelNotFoundError as me:
            st.error(str(me))
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")


# ==================================================================
# PAGE 2 -- COMPLAINT DASHBOARD
# ==================================================================
elif page == "Complaint Dashboard":
    st.title("Complaint Dashboard")

    try:
        records = database.fetch_all_complaints()
    except Exception as e:
        st.error(f"Could not load complaints from the database: {e}")
        records = []

    if not records:
        st.info("No complaints have been submitted yet. Go to 'Complaint Submission' to add one.")
    else:
        df = pd.DataFrame(records)
        df["confidence_pct"] = (df["confidence"] * 100).round(1)

        # ---- Summary metrics ----
        st.subheader("Summary")
        total = len(df)
        payments_count = df["department"].str.contains("Payments", case=False, na=False).sum()
        it_count = (df["department"] == "IT Support").sum()
        fraud_count = (df["department"] == "Fraud & Security").sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Complaints", total)
        m2.metric("Payments-related", int(payments_count))
        m3.metric("IT Support", int(it_count))
        m4.metric("Fraud & Security", int(fraud_count))

        st.markdown("---")

        # ---- Filters ----
        st.subheader("Filters")
        f1, f2, f3 = st.columns(3)
        dept_options = ["All"] + sorted(df["department"].unique().tolist())
        cat_options = ["All"] + sorted(df["predicted_category"].unique().tolist())
        status_options = ["All"] + sorted(df["status"].unique().tolist())

        selected_dept = f1.selectbox("Department", dept_options)
        selected_cat = f2.selectbox("Category", cat_options)
        selected_status = f3.selectbox("Status", status_options)

        filtered = df.copy()
        if selected_dept != "All":
            filtered = filtered[filtered["department"] == selected_dept]
        if selected_cat != "All":
            filtered = filtered[filtered["predicted_category"] == selected_cat]
        if selected_status != "All":
            filtered = filtered[filtered["status"] == selected_status]

        st.markdown("---")

        # ---- Charts ----
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Complaints by Department**")
            dept_counts = filtered["department"].value_counts()
            if len(dept_counts) > 0:
                st.bar_chart(dept_counts)
            else:
                st.caption("No data for current filter.")
        with c2:
            st.markdown("**Complaints by Category**")
            cat_counts = filtered["predicted_category"].value_counts()
            if len(cat_counts) > 0:
                st.bar_chart(cat_counts)
            else:
                st.caption("No data for current filter.")

        st.markdown("---")

        # ---- Table ----
        st.subheader(f"Complaint Records ({len(filtered)})")
        display_df = filtered[[
            "complaint_id", "complaint_text", "predicted_category",
            "confidence_pct", "department", "status", "timestamp"
        ]].rename(columns={
            "complaint_id": "Complaint ID",
            "complaint_text": "Complaint Text",
            "predicted_category": "Category",
            "confidence_pct": "Confidence (%)",
            "department": "Department",
            "status": "Status",
            "timestamp": "Timestamp",
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ---- Status update ----
        st.markdown("---")
        st.subheader("Update Complaint Status")
        s1, s2, s3 = st.columns([2, 2, 1])
        complaint_ids = filtered["complaint_id"].tolist()
        if complaint_ids:
            selected_id = s1.selectbox("Complaint ID", complaint_ids)
            new_status = s2.selectbox("New Status", STATUS_OPTIONS)
            if s3.button("Update Status"):
                try:
                    database.update_status(selected_id, new_status)
                    st.success(f"Status for {selected_id} updated to '{new_status}'.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not update status: {e}")
        else:
            st.caption("No complaints match the current filter to update.")


# ==================================================================
# PAGE 3 -- MODEL PERFORMANCE
# ==================================================================
elif page == "Model Performance":
    st.title("Model Performance")

    metrics = load_metrics()

    if metrics is None:
        st.error(
            "Metrics file not found. Please run 'python src/train_model.py' "
            "to train the model and generate evaluation metrics."
        )
    else:
        st.markdown(
            "These numbers come from actually training and evaluating the model "
            "on a held-out test set (20% of the dataset, not used during training). "
            "Nothing here is hardcoded."
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
        m2.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
        m3.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
        m4.metric("F1-score", f"{metrics['f1_score'] * 100:.1f}%")

        st.caption(
            f"Training samples: {metrics['n_train_samples']}  |  "
            f"Testing samples: {metrics['n_test_samples']}  |  "
            f"Total samples: {metrics['n_total_samples']}"
        )

        # Honest disclosure about the stricter template-holdout evaluation
        strict = metrics.get("stricter_template_holdout_eval")
        if strict:
            with st.expander("A stricter, more honest evaluation (recommended reading)"):
                st.markdown(
                    "The dataset used in this prototype is synthetically generated "
                    "from sentence templates (with varied filler words, openers/closers, "
                    "and typos). A standard random train/test split can place near-duplicate "
                    "sentences (same base template) on both sides of the split, which "
                    "inflates accuracy. To be transparent, we also evaluated the model on a "
                    "**stricter split where the test set uses complaint templates the model "
                    "never saw during training at all**:"
                )
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Accuracy", f"{strict['accuracy'] * 100:.1f}%")
                sc2.metric("Precision", f"{strict['precision'] * 100:.1f}%")
                sc3.metric("Recall", f"{strict['recall'] * 100:.1f}%")
                sc4.metric("F1-score", f"{strict['f1_score'] * 100:.1f}%")
                st.caption(strict.get("note", ""))

        st.markdown("---")

        # Confusion matrix
        st.subheader("Confusion Matrix")
        if os.path.exists(CONFUSION_MATRIX_PATH):
            st.image(CONFUSION_MATRIX_PATH, use_container_width=True)
        else:
            st.warning("Confusion matrix image not found. Re-run training to generate it.")

        st.markdown("---")

        # Per-class report
        st.subheader("Per-Category Classification Report")
        report = metrics.get("classification_report", {})
        rows = []
        for cat in metrics.get("categories", []):
            if cat in report:
                rows.append({
                    "Category": cat,
                    "Precision": round(report[cat]["precision"], 3),
                    "Recall": round(report[cat]["recall"], 3),
                    "F1-score": round(report[cat]["f1-score"], 3),
                    "Support": int(report[cat]["support"]),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("---")

        # Model comparison
        comparison = metrics.get("model_comparison")
        if comparison:
            st.subheader("Model Comparison (for reference)")
            st.caption(
                "Logistic Regression is the model actually used by this application. "
                "Naive Bayes and Linear SVM are shown only for academic comparison."
            )
            comp_df = pd.DataFrame(comparison).T
            comp_df = comp_df.rename(columns={
                "accuracy": "Accuracy", "precision": "Precision",
                "recall": "Recall", "f1_score": "F1-score",
            })
            for col in comp_df.columns:
                comp_df[col] = (comp_df[col] * 100).round(2).astype(str) + "%"
            st.dataframe(comp_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Supported Complaint Categories")
        st.write(", ".join(ALL_CATEGORIES))

        st.markdown("---")
        st.subheader("How the model works")
        st.markdown(
            "- **TF-IDF (Term Frequency-Inverse Document Frequency)** converts complaint "
            "text into numerical features based on how important each word/phrase is "
            "to a complaint, relative to the whole dataset.\n"
            "- **Logistic Regression** learns the relationship between those numerical "
            "features and the complaint categories from labelled training examples.\n"
            "- **The routing layer** (a separate rule-based lookup, not part of the ML "
            "model) maps the predicted category to the responsible department."
        )


# ==================================================================
# PAGE 4 -- ABOUT PROJECT
# ==================================================================
else:
    st.title("About This Project")

    st.header("Problem Statement")
    st.write(
        "Customer complaints are often manually reviewed and forwarded to the "
        "appropriate department. This can take time and may result in incorrect "
        "routing. The proposed system uses NLP and machine learning to automatically "
        "classify customer complaints and route them to the appropriate department."
    )

    st.header("Objectives")
    st.markdown(
        "- Automate complaint classification.\n"
        "- Reduce manual routing effort.\n"
        "- Improve consistency of complaint categorization.\n"
        "- Route complaints to appropriate departments.\n"
        "- Maintain a centralized complaint record.\n"
        "- Demonstrate practical use of NLP in customer service."
    )

    st.header("System Architecture")
    st.code(
        "Customer Complaint\n"
        "        |\n"
        "Text Preprocessing (cleaning)\n"
        "        |\n"
        "NLP Classification Model (TF-IDF + Logistic Regression)\n"
        "        |\n"
        "Complaint Category\n"
        "        |\n"
        "Department Mapping (rule-based routing.py)\n"
        "        |\n"
        "Automatic Routing\n"
        "        |\n"
        "Department Queue / Dashboard (SQLite + Streamlit)",
        language=None,
    )

    st.header("Technologies Used")
    st.markdown(
        "- **Python** -- core programming language\n"
        "- **Pandas / NumPy** -- data handling\n"
        "- **Scikit-learn** -- TF-IDF vectorization, Logistic Regression, evaluation\n"
        "- **Streamlit** -- interactive web dashboard\n"
        "- **SQLite** -- lightweight, file-based complaint storage\n"
        "- **Matplotlib / Seaborn** -- confusion matrix visualization"
    )

    st.header("Two-Stage Design")
    st.markdown(
        "This system deliberately separates two concerns:\n\n"
        "**Stage 1 -- NLP Classification:** Given raw complaint text, predict the "
        "most likely complaint category (e.g. *'Duplicate Transaction'*). This is "
        "the only job of the machine learning model.\n\n"
        "**Stage 2 -- Routing:** Given a category, look up which department is "
        "responsible for it (e.g. *Duplicate Transaction -> Payments Department*). "
        "This is a simple, explicit rule-based mapping -- not something the ML "
        "model 'knows'. Keeping this separate makes the system transparent, easy "
        "to audit, and easy to reconfigure if departments change."
    )

    st.header("Limitations")
    st.markdown(
        "- The dataset is synthetically generated from sentence templates (with "
        "varied filler words, phrasing, and simulated typos) rather than sourced "
        "from real customer complaints, so reported accuracy on a random split may "
        "be optimistic (see the stricter evaluation on the Model Performance page).\n"
        "- The model only supports the 12 categories it was trained on and will "
        "always pick the closest one, even for out-of-scope complaints.\n"
        "- Confidence scores reflect the model's internal probability estimate, "
        "not a guarantee of correctness.\n"
        "- The routing logic is a simple static mapping and does not account for "
        "department workload, priority, or escalation rules."
    )

    st.header("Future Scope")
    st.markdown(
        "- Train on real, anonymised customer complaint data for better generalisation.\n"
        "- Add multi-label classification for complaints that span multiple categories.\n"
        "- Add priority/urgency detection to triage critical complaints faster.\n"
        "- Add authentication and role-based access for department staff.\n"
        "- Integrate with a real ticketing system / email notifications."
    )
