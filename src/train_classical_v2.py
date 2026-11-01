
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed_v2"
MODELS = ROOT / "models"

MODELS.mkdir(parents=True, exist_ok=True)

# Load the already-preprocessed train/test datasets.
X_train = pd.read_csv(DATA / "X_train.csv")
X_test = pd.read_csv(DATA / "X_test.csv")

y_train = pd.read_csv(DATA / "y_train.csv").iloc[:, 0]
y_test = pd.read_csv(DATA / "y_test.csv").iloc[:, 0]

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("Training class distribution:")
print(y_train.value_counts().sort_index())

# Cross-validation is performed on the training set only.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

# Tune both models using the same primary selection metric.
searches = {
    "Logistic Regression": GridSearchCV(
        estimator=LogisticRegression(
            max_iter=5000,
            random_state=42,
        ),
        param_grid={
            "C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "solver": ["liblinear", "lbfgs"],
            "class_weight": [None, "balanced"],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    ),
    "SVM": GridSearchCV(
        estimator=SVC(
            probability=True,
            random_state=42,
        ),
        param_grid=[
            {
                "kernel": ["rbf"],
                "C": [0.1, 0.5, 1, 2, 5, 10, 20],
                "gamma": ["scale", "auto", 0.01, 0.1, 1],
                "class_weight": [None, "balanced"],
            },
            {
                "kernel": ["linear"],
                "C": [0.1, 0.5, 1, 2, 5, 10],
                "class_weight": [None, "balanced"],
            },
        ],
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    ),
}

results = []

for name, search in searches.items():
    print(f"\n{'=' * 60}")
    print(f"Tuning {name}...")

    # Hyperparameter selection uses training data only.
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Evaluate the selected model once on the held-out test set.
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]

    row = {
        "Model": name,
        "CV_ROC_AUC": search.best_score_,
        "Test_Accuracy": accuracy_score(y_test, predictions),
        "Test_Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Test_Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "Test_F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "Test_ROC_AUC": roc_auc_score(y_test, probabilities),
    }
    results.append(row)

    print("Best parameters:", search.best_params_)
    print(f"Best CV ROC-AUC: {search.best_score_:.3f}")

    for metric, value in row.items():
        if metric not in ("Model", "CV_ROC_AUC"):
            print(f"{metric}: {value:.3f}")

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No disease label", "Disease label"],
            zero_division=0,
        )
    )

    filename = (
        "logistic_regression_v2.joblib"
        if name == "Logistic Regression"
        else "svm_rbf_v2.joblib"
    )

    joblib.dump(best_model, MODELS / filename)
    print(f"Saved: {MODELS / filename}")

results_df = pd.DataFrame(results)
results_path = ROOT / "classical_model_results_v2.csv"
results_df.to_csv(results_path, index=False)

print("\nFINAL MODEL COMPARISON")
print(results_df.round(3).to_string(index=False))
print(f"\nResults saved to: {results_path}")
