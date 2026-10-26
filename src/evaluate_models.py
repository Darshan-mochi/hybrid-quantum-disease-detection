
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
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
DATA_PATH = ROOT / "data" / "processed.cleveland.data"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "data" / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target",
]
FEATURES = ["age", "trestbps", "chol", "thalach"]

df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=COLUMNS,
    na_values="?",
)
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=["target"])

X = df[FEATURES]
y = (df["target"] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train_encoded = scaler.fit_transform(X_train_imp)
X_test_encoded = scaler.transform(X_test_imp)

# Refit the classical baseline using this exact split.
classical = LogisticRegression(
    max_iter=2000,
    random_state=42,
)

start = time.perf_counter()
classical.fit(X_train_encoded, y_train)
classical_seconds = time.perf_counter() - start

classical_pred = classical.predict(X_test_encoded)
classical_prob = classical.predict_proba(X_test_encoded)[:, 1]

def summarize(name, y_true, pred, prob, seconds):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(
            y_true, pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, pred, zero_division=0
        ),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
        "training_seconds": seconds,
    }

results = [
    summarize(
        "Logistic Regression (4 features)",
        y_test,
        classical_pred,
        classical_prob,
        classical_seconds,
    )
]

print("CLASSICAL MODEL")
print(classification_report(y_test, classical_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, classical_pred))

# The saved quantum weights are not, by themselves, a complete
# classifier. Evaluate VQC after reconstructing the same circuit.
weights_path = MODELS_DIR / "vqc_4qubits_weights.npy"

if weights_path.exists():
    from qiskit.circuit.library import (
        z_feature_map,
        real_amplitudes,
    )
    from qiskit_machine_learning.algorithms import VQC
    from qiskit_machine_learning.optimizers import COBYLA

    feature_map = z_feature_map(
        feature_dimension=4,
        reps=1,
    )
    ansatz = real_amplitudes(
        num_qubits=4,
        reps=1,
    )

    initial_weights = np.load(weights_path)

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=100),
        initial_point=initial_weights,
    )

    # Refit the circuit on the training split for a reproducible
    # evaluation procedure. This is a new training run.
    start = time.perf_counter()
    vqc.fit(X_train_encoded, y_train.to_numpy())
    quantum_seconds = time.perf_counter() - start

    quantum_pred = np.asarray(
        vqc.predict(X_test_encoded)
    ).reshape(-1).astype(int)

    quantum_prob = np.asarray(
        vqc.predict_proba(X_test_encoded)
    )

    if quantum_prob.ndim == 2:
        quantum_prob = quantum_prob[:, 1]

    results.append(
        summarize(
            "Quantum VQC (4 qubits)",
            y_test,
            quantum_pred,
            quantum_prob,
            quantum_seconds,
        )
    )

    print("\nQUANTUM MODEL")
    print(classification_report(y_test, quantum_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, quantum_pred))
else:
    print(
        "\nSaved VQC weights were not found. "
        "Only the classical model was evaluated."
    )

results_df = pd.DataFrame(results)
output_file = OUTPUT_DIR / "evaluation_results.csv"
results_df.to_csv(output_file, index=False)

print("\nEVALUATION SUMMARY")
print(results_df.round(3).to_string(index=False))
print(f"\nSaved results to: {output_file}")
