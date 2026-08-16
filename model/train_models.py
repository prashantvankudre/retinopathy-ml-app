"""
train_models.py
----------------
Trains 6 classification models on the UCI Diabetic Retinopathy Debrecen
dataset with hyperparameter tuning (GridSearchCV, 5-fold stratified CV),
evaluates each on a held-out test split with 6 metrics, and generates SHAP
feature-importance plots for interpretability.

Dataset source: UCI ML Repository, id=329
https://archive.ics.uci.edu/dataset/329/diabetic+retinopathy+debrecen

Note: this dataset has no patient_id column, so a single stratified
train/test split (not a group-aware split) is appropriate here. The
group-aware path below is kept for reuse on datasets that DO carry a
patient/group identifier, but is inactive on this dataset.

Outputs:
    - model/<name>.pkl                  -> best tuned sklearn Pipeline
    - results/metrics_comparison.csv    -> comparison table used in README + app
    - results/shap_<model>.png          -> SHAP feature-importance plot per model
    - results/feature_columns.json
    - test_data.csv                     -> held-out test split (features + class)

Run with:  python model/train_models.py
"""

import json
import logging
import os
from builtins import open

import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedKFold,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "class"
GROUP_COL = "patient_id"  # not present in this dataset; kept for reuse

# SHAP's KernelExplainer is O(n^2)-ish; cap sample sizes so it stays fast
SHAP_BACKGROUND_SIZE = 50
SHAP_EXPLAIN_SIZE = 100

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "retinopathy.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TEST_DATA_PATH = os.path.join(BASE_DIR, "test_data.csv")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "train_data.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_data():
    """Load dataset, report basic diagnostics, and remove duplicates if present."""
    df = pd.read_csv(DATA_PATH)
    log.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        log.info(f"Dropped {dup_count} duplicate row(s) -> {df.shape[0]} rows remain")

    log.info(f"Missing values: {df.isnull().sum().sum()}")
    log.info(f"Class distribution:\n{df[TARGET_COL].value_counts()}")

    class_counts = df[TARGET_COL].value_counts(normalize=True)
    minority_ratio = class_counts.min()

    if minority_ratio < 0.10:
        log.warning(
            f"Severe class imbalance detected: minority class = {minority_ratio:.3f}. "
            "Consider using class weights, SMOTE, or calibration."
        )
    elif minority_ratio < 0.20:
        log.info(
            f"Moderate class imbalance detected: minority class = {minority_ratio:.3f}. "
            "Stratified CV is recommended."
        )

    return df


def split_data(df, X, y):
    """Group-aware split if a group/patient identifier column exists
    (prevents leakage from repeated records of the same subject);
    otherwise a standard stratified train/test split.
    """
    if GROUP_COL in df.columns:
        log.info(f"'{GROUP_COL}' column detected -> using GroupShuffleSplit")
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X, y, groups=df[GROUP_COL]))
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

    log.info(f"No '{GROUP_COL}' column -> using stratified train_test_split")
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def build_models():
    """Returns dict of {name: (pipeline, hyperparameter grid)}."""
    return {
        "Logistic Regression": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
            {"clf__C": [0.1, 1, 10], "clf__penalty": ["l2"]},
        ),
        "Decision Tree": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
            {"clf__max_depth": [5, 7, 10], "clf__min_samples_split": [2, 5, 10]},
        ),
        "kNN": (
            Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [5, 9, 11, 15]},
        ),
        "Naive Bayes": (
            Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]),
            {},  # no hyperparameters worth tuning
        ),
        "Random Forest": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", RandomForestClassifier(random_state=RANDOM_STATE))]),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [8, 10, None]},
        ),
        "SVM": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))]),
            {"clf__C": [0.5, 1, 2], "clf__gamma": ["scale", "auto"]},
        ),
    }


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def generate_shap(pipeline, X_train, model_name):
    """Save a SHAP summary plot for the tuned model.

    IMPORTANT: the classifier was fit on SCALED features, so the explainer
    must also be given scaled data - explaining it with raw, unscaled
    features (as in the original draft) silently produces misleading
    attributions. Tree/linear models get fast, exact explainers; kNN/SVM/NB
    fall back to KernelExplainer, which is expensive, so we cap the
    background and explain-set sizes to keep runtime reasonable.
    """
    if not SHAP_AVAILABLE:
        log.warning(f"SHAP not installed - skipping SHAP plot for {model_name}")
        return

    try:
        scaler = pipeline.named_steps["scaler"]
        clf = pipeline.named_steps["clf"]
        X_train_scaled = pd.DataFrame(
            scaler.transform(X_train), columns=X_train.columns, index=X_train.index
        )

        if isinstance(clf, (DecisionTreeClassifier, RandomForestClassifier)):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_train_scaled)
        elif isinstance(clf, LogisticRegression):
            explainer = shap.LinearExplainer(clf, X_train_scaled)
            shap_values = explainer.shap_values(X_train_scaled)
        else:
            # kNN, SVM, Naive Bayes: no closed-form explainer -> KernelExplainer.
            # Cap sizes; this is still the slowest path.
            background = shap.kmeans(X_train_scaled, min(SHAP_BACKGROUND_SIZE, len(X_train_scaled)))
            explain_sample = X_train_scaled.sample(
                n=min(SHAP_EXPLAIN_SIZE, len(X_train_scaled)), random_state=RANDOM_STATE
            )
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(explain_sample)
            X_train_scaled = explain_sample  # for the summary plot below

        plt.figure(figsize=(10, 6))
        vals = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap.summary_plot(vals, X_train_scaled, show=False)
        out_path = os.path.join(RESULTS_DIR, f"shap_{model_name}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        log.info(f"Saved SHAP plot -> {out_path}")

    except Exception as e:
        log.warning(f"SHAP failed for {model_name}: {e}")


def main():
    df = load_data()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = split_data(df, X, y)
    log.info(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    models = build_models()
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, (pipeline, param_grid) in models.items():
        log.info(f"Training {name} (5-fold CV hyperparameter search)...")
        grid = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        
        log.info(f"Best params for {name}: {grid.best_params_}")
        log.info(f"Best CV ROC-AUC for {name}: {grid.best_score_:.4f}")

        metrics = evaluate(best_model, X_test, y_test)
        results[name] = metrics
        log.info(f"Held-out test metrics: {json.dumps({k: round(v, 4) for k, v in metrics.items()})}")

        safe_name = name.lower().replace(" ", "_")
        model_path = os.path.join(MODEL_DIR, f"{safe_name}.pkl")
        joblib.dump(best_model, model_path)
        log.info(f"Saved model -> {model_path}")

        generate_shap(best_model, X_train, safe_name)

    results_df = pd.DataFrame(results).T.round(4)
    results_df.index.name = "Model"
    results_df.to_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"))
    log.info("Saved metrics_comparison.csv")
    print(results_df)

    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    log.info(f"Saved train_data.csv ({train_df.shape[0]} rows)")

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values
    test_df.to_csv(TEST_DATA_PATH, index=False)
    log.info(f"Saved test_data.csv ({test_df.shape[0]} rows)")

    with open(os.path.join(RESULTS_DIR, "feature_columns.json"), "w") as f:
        json.dump(list(X.columns), f)
    log.info("Saved feature_columns.json")


if __name__ == "__main__":
    main()
