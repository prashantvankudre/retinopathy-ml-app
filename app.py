"""
Diabetic Retinopathy Classification - Interactive Streamlit Demo
-------------------------------------------------------------------
Upload the provided test_data.csv (or your own CSV with the same 18 feature
columns + a 'class' column), pick a trained model (or "All Models" to
compare all six at once), and see evaluation metrics, confusion matrix,
and classification report computed live on YOUR uploaded data.
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --------------------------------------------------------------------------
# Config / paths
# --------------------------------------------------------------------------
st.set_page_config(page_title="Diabetic Retinopathy Classifier", page_icon="\U0001F441\uFE0F", layout="wide",initial_sidebar_state="expanded")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DEFAULT_TEST_DATA = os.path.join(BASE_DIR, "test_data.csv")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
    "SVM": "svm.pkl",
}
ALL_MODELS_OPTION = "All Models (Compare)"

TARGET_COL = "class"

# Primary metric used to pick a "best model" — MCC is the most robust
# single metric for binary classification since it accounts for all four
# confusion-matrix cells and isn't distorted by class imbalance, unlike
# raw Accuracy. The full table is still shown so you can judge by any
# metric that matters more for your use case (e.g. Recall for screening).
RANKING_METRIC = "MCC"

FEATURE_DESCRIPTIONS = {
    "quality": "Binary: The binary result of quality assessment. 0 = bad quality 1 = sufficient quality.",
    "pre_screening": "Binary: The binary result of pre-screening, where 1 indicates severe retinal abnormality and 0 its lack.",
    "ma1": "ma1 - ma-6 contain the results of MA detection. Each feature value stand for the number of MAs found at the confidence levels alpha = 0.5, . . . , 1, respectively.",
    "ma2": "Microaneurysm detections at confidence level 2",
    "ma3": "Microaneurysm detections at confidence level 3",
    "ma4": "Microaneurysm detections at confidence level 4",
    "ma5": "Microaneurysm detections at confidence level 5",
    "ma6": "Microaneurysm detections at confidence level 6",
    "exudate1": "exudate1 - exudate8 contain the same information as 2-7) for exudates. However, as exudates are represented by a set of points rather than the number of pixels constructing the lesions, these features are normalized by dividing the number of lesions with the diameter of the ROI to compensate different image sizes. (normalized)",
    "exudate2": "Exudate detection measure 2 (normalized)",
    "exudate3": "Exudate detection measure 3 (normalized)",
    "exudate5": "Exudate detection measure 5 (normalized)",
    "exudate6": "Exudate detection measure 6 (normalized)",
    "exudate7": "Exudate detection measure 7 (normalized)",
    "exudate8": "Exudate detection measure 8 (normalized)",
    "macula_opticdisc_distance": "Euclidean distance: The euclidean distance of the center of the macula and the center of the optic disc to provide important information regarding the patient's condition. This feature is also normalized with the diameter of the ROI.",
    "opticdisc_diameter": "Diameter of the optic disc (normalized)",
    "am_fm_classification": "Binary result of AM/FM-based texture classification",
}


@st.cache_resource
def load_model(model_filename):
    path = os.path.join(MODEL_DIR, model_filename)
    return joblib.load(path)


@st.cache_resource
def load_all_models():
    return {name: load_model(fname) for name, fname in MODEL_FILES.items()}


@st.cache_data
def load_feature_columns():
    path = os.path.join(RESULTS_DIR, "feature_columns.json")
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_default_test_data():
    return pd.read_csv(DEFAULT_TEST_DATA)


@st.cache_data
def load_training_comparison():
    return pd.read_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"), index_col="Model")


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def render_confusion_and_report(y_true, y_pred, model_label):
    col_a, col_b = st.columns([1, 1])
    with col_a:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No DR", "DR"], yticklabels=["No DR", "DR"], ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {model_label}")
        st.pyplot(fig)
        plt.close(fig)
    with col_b:
        report = classification_report(y_true, y_pred, target_names=["No DR", "DR"])
        st.text("Classification Report")
        st.code(report)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("\U0001F441\uFE0F Diabetic Retinopathy Classifier")
st.sidebar.markdown(
    "Predict presence of diabetic retinopathy from retinal image-derived "
    "features using 6 trained classical ML models.\n\n"
    "Dataset source: [UCI Diabetic Retinopathy Debrecen]"
    "(https://archive.ics.uci.edu/dataset/329/diabetic+retinopathy+debrecen)"
)

st.sidebar.subheader("1. Upload test data (CSV)")
uploaded_file = st.sidebar.file_uploader(
    "CSV with the 18 feature columns + 'class' column",
    type=["csv"],
)

st.sidebar.subheader("2. Choose a model")
model_choice = st.sidebar.selectbox(
    "Model", [ALL_MODELS_OPTION] + list(MODEL_FILES.keys())
)

with st.sidebar.expander("Feature reference"):
    for feat, desc in FEATURE_DESCRIPTIONS.items():
        st.markdown(f"**{feat}** — {desc}")


st.sidebar.subheader("LINKS:")
st.sidebar.markdown("[README](https://github.com/yourname/yourrepo/blob/main/README.md)")
st.sidebar.markdown("[GitHub Repository](https://github.com/yourname/yourrepo/blob/main/README.md)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("Diabetic Retinopathy Detection — Model Comparison Dashboard")
st.caption(
    "Dataset: UCI Diabetic Retinopathy Debrecen (1,146 unique patient records "
    "after de-duplication, 18 features extracted from Messidor retinal fundus "
    "images, binary target)."
)

feature_cols = load_feature_columns()

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success(f"Loaded uploaded file with {data.shape[0]} rows.")
else:
    data = load_default_test_data()
    st.info(
        f"No file uploaded — using the bundled `test_data.csv` "
        f"({data.shape[0]} held-out rows) as a demo."
    )

missing_cols = [c for c in feature_cols if c not in data.columns]
if missing_cols:
    st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
    st.stop()

extra_cols = [c for c in data.columns if c not in feature_cols + [TARGET_COL]]
if extra_cols:
    st.warning(f"Extra columns ignored: {extra_cols}")

has_target = TARGET_COL in data.columns
X = data[feature_cols]

st.subheader("Preview of data")
st.dataframe(data.head(10), use_container_width=True)

# ==========================================================================
# BRANCH 1: "All Models (Compare)" — run all 6 models on the uploaded data
# ==========================================================================
if model_choice == ALL_MODELS_OPTION:
    models = load_all_models()

    predictions = {}
    probabilities = {}
    for name, mdl in models.items():
        predictions[name] = mdl.predict(X)
        probabilities[name] = mdl.predict_proba(X)[:, 1]

    if has_target:
        y_true = data[TARGET_COL]

        st.subheader("Live Model Comparison on Your Uploaded Data")
        live_results = {
            name: compute_metrics(y_true, predictions[name], probabilities[name])
            for name in models
        }
        live_df = pd.DataFrame(live_results).T.round(4)
        live_df.index.name = "Model"
        live_df = live_df.sort_values(RANKING_METRIC, ascending=False)

        best_model_name = live_df.index[0]
        best_score = live_df.loc[best_model_name, RANKING_METRIC]

        st.success(
            f"\U0001F3C6 **Best model on this data: {best_model_name}** "
            f"(ranked by {RANKING_METRIC} = {best_score:.4f}, the most robust "
            f"single metric for binary classification)"
        )

        st.dataframe(
            live_df.style.highlight_max(axis=0, color="lightgreen"),
            use_container_width=True,
        )

        fig, ax = plt.subplots(figsize=(9, 4))
        live_df[["Accuracy", "AUC", "F1", "MCC"]].plot(kind="bar", ax=ax)
        ax.set_ylabel("Score")
        ax.set_title("All 6 Models — Performance on Your Uploaded Data")
        ax.legend(loc="lower right")
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Per-Model Confusion Matrix & Classification Report")
        for name in live_df.index:  # show best-first
            with st.expander(f"{name}  (MCC = {live_df.loc[name, 'MCC']:.4f})"):
                render_confusion_and_report(y_true, predictions[name], name)

    else:
        st.warning(
            "Uploaded data has no 'class' column, so models can't be scored "
            "or ranked against ground truth. Showing side-by-side predictions "
            "from all 6 models instead."
        )
        preds_df = data[feature_cols[:3]].copy()
        for name in models:
            preds_df[f"{name} — prediction"] = predictions[name]
            preds_df[f"{name} — risk prob."] = np.round(probabilities[name], 3)
        st.dataframe(preds_df.head(15), use_container_width=True)

        # Simple agreement signal even without ground truth: how often do
        # all 6 models agree on the predicted class for a given row?
        pred_matrix = pd.DataFrame(predictions)
        agreement = (pred_matrix.nunique(axis=1) == 1).mean()
        st.caption(
            f"All 6 models agree on the same prediction for "
            f"{agreement * 100:.1f}% of rows (no accuracy/ranking claim — "
            f"just a consistency signal, since there's no ground truth here)."
        )

    st.download_button(
        "Download all models' predictions as CSV",
        pd.DataFrame({
            **{f"{n}_prediction": predictions[n] for n in models},
            **{f"{n}_risk_probability": np.round(probabilities[n], 3) for n in models},
        }).to_csv(index=False).encode("utf-8"),
        file_name="all_models_predictions.csv",
        mime="text/csv",
    )

# ==========================================================================
# BRANCH 2: single model selected
# ==========================================================================
else:
    model = load_model(MODEL_FILES[model_choice])
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    result_df = data.copy()
    result_df["prediction"] = y_pred
    result_df["risk_probability"] = np.round(y_proba, 3)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"Evaluation metrics — {model_choice}")
        if has_target:
            y_true = data[TARGET_COL]
            metrics = compute_metrics(y_true, y_pred, y_proba)
            metrics_df = pd.DataFrame(
                {"Metric": list(metrics.keys()), "Value": [round(v, 4) for v in metrics.values()]}
            )
            st.table(metrics_df.set_index("Metric"))
            st.caption(
                "Tip: select **All Models (Compare)** from the dropdown to see "
                "which model performs best on this specific uploaded data."
            )
        else:
            st.warning(
                "Uploaded data has no 'class' column, so evaluation metrics "
                "can't be computed. Predictions are still shown on the right."
            )

    with col2:
        st.subheader("Predictions")
        st.dataframe(
            result_df[feature_cols[:3] + ["prediction", "risk_probability"]].head(10),
            use_container_width=True,
        )
        st.download_button(
            "Download full predictions as CSV",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

    if has_target:
        st.subheader("Confusion Matrix & Classification Report")
        render_confusion_and_report(y_true, y_pred, model_choice)

# --------------------------------------------------------------------------
# Reference table: performance on the ORIGINAL held-out test split
# (from training time — static, not the uploaded data above)
# --------------------------------------------------------------------------
st.divider()
st.subheader("Reference: All 6 Models on the Original Training-Time Test Split")
st.caption(
    "This table reflects each model's performance on the held-out split used "
    "during training (see README for full methodology) — kept here for "
    "reference alongside the live results on your uploaded data above."
)
comparison_df = load_training_comparison()
st.dataframe(comparison_df.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)