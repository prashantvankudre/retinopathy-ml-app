# Diabetic Retinopathy Classification — Multi-Model ML App

## a. Problem Statement

Diabetic retinopathy (DR) is a leading cause of preventable blindness among
working-age adults worldwide, and it is typically screened for by manually
grading retinal fundus photographs — a process that is time-consuming and
requires specialist ophthalmologist review. Automated pre-screening from
image-derived features can help flag likely-positive cases for priority
review, extending screening capacity in resource-constrained settings.

This project frames DR detection as a **binary classification problem**:
given a set of quantitative features automatically extracted from a
patient's retinal fundus image (microaneurysm counts, exudate measures,
optic disc geometry, etc.), predict whether the image shows signs of
diabetic retinopathy (`class = 1`) or not (`class = 0`).

Six classical machine learning classifiers are trained on the same dataset,
evaluated with six standard metrics, and made available for interactive
comparison through a Streamlit web application. From a pharma / clinical
research standpoint, this mirrors a realistic screening-triage question:
among models built on the same fixed feature set, which gives the best
balance of sensitivity (Recall — catching true DR cases) against overall
reliability (Precision, MCC), and how much value do more complex models
(SVM, ensembles) actually add over a simple, interpretable baseline?

## b. Dataset Description

- **Source:** [UCI Machine Learning Repository — Diabetic Retinopathy
  Debrecen Data Set](https://archive.ics.uci.edu/dataset/329/diabetic+retinopathy+debrecen)
  (Antal & Hajdu, 2014, University of Debrecen, Hungary). Features were
  extracted from the publicly available **Messidor** retinal fundus image
  set by an automated screening/lesion-detection algorithm.
- **Instances:** 1,151 patient image records (≥ 500 required ✅); 1,146 after
  removing 5 exact duplicate rows.
- **Features:** 18 numeric/binary features (≥ 12 required ✅)
- **Target:** `class` — 1 = signs of diabetic retinopathy present, 0 = absent
  (binary classification)
- **Class balance:** 606 positive / 540 negative — reasonably balanced
- **Missing values:** none

| Feature | Description |
|---|---|
| quality | Binary image quality assessment (1 = good quality) |
| pre_screening | Binary pre-screening result (1 = severe retinal abnormality flagged) |
| ma1 – ma6 | Microaneurysm detection counts at 6 increasing confidence thresholds |
| exudate1, exudate2, exudate3, exudate5, exudate6, exudate7, exudate8 | Normalized exudate-detection measures at several confidence levels |
| macula_opticdisc_distance | Euclidean distance between macula center and optic disc center (normalized by ROI diameter) |
| opticdisc_diameter | Diameter of the optic disc (normalized) |
| am_fm_classification | Binary result of an AM/FM texture-based classifier |

Unlike raw fundus photographs, these are **pre-extracted tabular features**
— this is what makes the dataset suitable for classical ML models (Logistic
Regression, Decision Tree, kNN, Naive Bayes, Random Forest, SVM) rather than
requiring a CNN/deep-learning image pipeline.

### Data quality note

Five exact duplicate rows were found and dropped before splitting (1,151 →
1,146 rows). This is a much smaller and less impactful duplication issue
than typically seen in other popular Kaggle medical datasets (e.g. the UCI
Heart Disease Kaggle mirror, which pads 302 unique patients up to 1,025 rows
via heavy duplication) — no special leakage-avoiding split strategy was
required here beyond the standard duplicate removal.

## c. GitHub Repository Link

`[https://github.com/prashantvankudre/retinopathy-ml-app]`

## d. Models Used

All six models were trained on the same 916-row training split and evaluated
on the same 230-row held-out test split (80/20, stratified by class).

Hyperparameters for each model were tuned with `GridSearchCV` (5-fold
stratified cross-validation, optimizing ROC-AUC) before final evaluation on
the held-out test split — see `model/train_models.py` for the exact grids
searched and the best parameters found per model.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.7435** | **0.8422** | **0.8621** | 0.6148 | 0.7177 | **0.5183** |
| Decision Tree | 0.6304 | 0.6913 | 0.6989 | 0.5328 | 0.6047 | 0.2782 |
| kNN | 0.5957 | 0.6710 | 0.6495 | 0.5164 | 0.5753 | 0.2037 |
| Naive Bayes | 0.6174 | 0.7006 | 0.6181 | **0.7295** | 0.6692 | 0.2272 |
| Random Forest (Ensemble) | 0.6478 | 0.7520 | 0.7204 | 0.5492 | 0.6233 | 0.3137 |
| SVM | 0.6826 | 0.7857 | 0.7882 | 0.5492 | 0.6473 | 0.3955 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | **Best overall model**, and tuning widened its lead further (best `C=10`) — highest Accuracy (0.74), AUC (0.84), Precision (0.86), and MCC (0.52). This suggests the relationship between these pre-extracted image features and DR presence is close to linearly separable, so the extra flexibility of non-linear models isn't adding much value here — and it's also the most interpretable choice for a clinical setting. |
| Decision Tree | Weakest model on almost every metric even after tuning `max_depth`/`min_samples_split` via grid search (best: depth 5, min split 10) — Accuracy 0.63, AUC 0.69. A single tree still overfits the training set's noise in these lesion-count features and doesn't generalize as smoothly as the ensemble or margin-based methods. |
| kNN | Also weak here (best `k=15`, AUC 0.67, MCC 0.20) — unlike on the Heart Disease dataset, "nearby" patients in this 18-dimensional feature space don't share diagnoses as reliably, likely because several features (the ma1–ma6 counts) are highly correlated with each other, distorting Euclidean distance. |
| Naive Bayes | Lowest Precision (0.62) but the **highest Recall (0.73)** of all six models — it over-predicts the positive class, which hurts Precision/MCC but means it catches the most true DR cases. Its independence assumption is clearly violated (microaneurysm counts at different confidence thresholds are strongly correlated with each other), which explains its middling overall performance despite the high Recall. Has no meaningful hyperparameters to tune. |
| Random Forest (Ensemble) | Middling results (Accuracy 0.65, AUC 0.75) even with tuning (best: 400 trees, unlimited depth) — better than a single Decision Tree as expected from ensembling, but still well behind Logistic Regression. With only ~916 training rows and 18 already-engineered features, there isn't much complex non-linear structure left for the forest to exploit beyond what a linear model already captures. |
| SVM | Second-best model after tuning (best `C=2`, `gamma='scale'`) — AUC 0.79, MCC 0.40. The RBF kernel picks up some non-linear structure that trees and kNN miss, but still falls short of plain Logistic Regression on this dataset. |

**Overall Winner for this dataset: Logistic Regression.** This is a useful,
slightly counterintuitive finding: the simplest, most interpretable model
outperformed every more complex alternative (Random Forest, SVM) on nearly
every metric, both before and after hyperparameter tuning. For a clinical
screening tool, that's a genuinely good outcome — it means the more
explainable model doesn't have to be traded off against raw performance. If
the priority were instead to minimize missed DR cases at any cost, Naive
Bayes' high Recall (0.73) would be worth revisiting despite its lower
overall MCC.

### Methodology notes

- **Hyperparameter tuning:** each model (except Naive Bayes, which has no
  meaningful hyperparameters) was tuned via `GridSearchCV` with 5-fold
  stratified cross-validation, optimizing ROC-AUC on the training split
  only — the held-out test split was never touched during tuning.
- **Explainability:** `model/train_models.py` optionally generates SHAP
  feature-importance plots per model (`results/shap_<model>.png`) when the
  `shap` package is installed (`pip install -r requirements-dev.txt`). This
  is a local/offline analysis step — the deployed Streamlit app does not
  depend on SHAP.
- **Leakage handling:** the script checks for a `patient_id`-style group
  column and would use a group-aware split to prevent the same subject's
  records spanning train/test; this dataset has no such column, so a
  standard stratified split is used (justified further above, under
  "Data quality note").

## Project Structure

```
retinopathy-ml-app/
├── app.py                        # Streamlit app (main entry point)
├── requirements.txt
├── README.md
├── retinopathy.csv                # Full original dataset (1,151 rows)
├── test_data.csv                  # Held-out test split used for the app demo
├── model/
│   ├── train_models.py            # Trains all 6 models, saves .pkl + metrics
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── svm.pkl
└── results/
    ├── metrics_comparison.csv     # 6-model x 6-metric comparison table
    └── feature_columns.json
```

## How to Run Locally

```bash
git clone https://github.com/prashantvankudre/retinopathy-ml-app.git
cd retinopathy-ml-app
pip install -r requirements.txt

# (Optional) Retrain all models from scratch:
python model/train_models.py

# Launch the app:
streamlit run app.py
```

## Live Streamlit App

`[https://retinopathy-ml-app.streamlit.app]`

## App Features

- **CSV upload:** upload a test CSV with the 18 feature columns (+ optional
  `class` column for evaluation) — or use the bundled `test_data.csv` demo
  data automatically if nothing is uploaded.
- **Model selection dropdown:** switch between all 6 trained models.
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1, MCC computed
  live on the uploaded data.
- **Confusion matrix & classification report:** visual + text breakdown of
  predictions vs. actual labels.
- **Model comparison table & chart:** side-by-side view of all 6 models'
  performance on the original held-out test split.

## Tech Stack

Python, scikit-learn, pandas, NumPy, Streamlit, Matplotlib, Seaborn.
