
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
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# 1. Locate the project folders
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

MODELS.mkdir(parents=True, exist_ok=True)


# 2. Load prepared training and testing data
X_train = pd.read_csv(DATA / "X_train.csv")
X_test = pd.read_csv(DATA / "X_test.csv")

y_train = pd.read_csv(DATA / "y_train.csv").iloc[:, 0]
y_test = pd.read_csv(DATA / "y_test.csv").iloc[:, 0]

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)


# 3. Define the models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),
    "SVM (RBF)": SVC(
        kernel="rbf",
        C=1.0,
        probability=True,
        random_state=42
    )
}


# 4. Train and evaluate each model
results = []

for name, model in models.items():
    print("\n" + "=" * 50)
    print("Model:", name)

    # Learn from training data
    model.fit(X_train, y_train)

    # Predict class labels for unseen test data
    y_pred = model.predict(X_test)

    # Estimate probability of class 1
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    auc = roc_auc_score(y_test, y_prob)

    # Display results
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 score:  {f1:.3f}")
    print(f"ROC-AUC:   {auc:.3f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["No disease", "Disease"],
            zero_division=0
        )
    )

    # Save the trained model
    filename = (
        "logistic_regression.joblib"
        if name == "Logistic Regression"
        else "svm_rbf.joblib"
    )
    joblib.dump(model, MODELS / filename)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": auc
    })


# 5. Compare the models in a table
results_df = pd.DataFrame(results)

print("\n" + "=" * 50)
print("CLASSICAL MODEL COMPARISON")
print(results_df.round(3).to_string(index=False))

results_df.to_csv(
    ROOT / "classical_model_results.csv",
    index=False
)

print("\nTrained models saved in:", MODELS)
print("Metrics saved to classical_model_results.csv")
