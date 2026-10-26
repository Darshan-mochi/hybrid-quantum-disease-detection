
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed_v2"
MODELS = ROOT / "models"

X_train = pd.read_csv(DATA / "X_train.csv")
X_test = pd.read_csv(DATA / "X_test.csv")
y_train = pd.read_csv(DATA / "y_train.csv").iloc[:, 0]
y_test = pd.read_csv(DATA / "y_test.csv").iloc[:, 0]

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, random_state=42
    ),
    "SVM RBF": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_test, probabilities)
    }
    results.append(row)

    print("\n", name)
    for metric, value in row.items():
        if metric != "Model":
            print(f"{metric}: {value:.3f}")

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    filename = (
        "logistic_regression_v2.joblib"
        if name == "Logistic Regression"
        else "svm_rbf_v2.joblib"
    )
    joblib.dump(model, MODELS / filename)

results_df = pd.DataFrame(results)
results_df.to_csv(
    ROOT / "classical_model_results_v2.csv",
    index=False
)

print("\nModel comparison:")
print(results_df.round(3).to_string(index=False))
print("\nSaved classical_model_results_v2.csv")
