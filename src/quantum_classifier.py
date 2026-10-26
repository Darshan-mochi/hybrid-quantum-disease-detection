
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from qiskit.circuit.library import z_feature_map, real_amplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA


# 1. Set project paths
ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "processed.cleveland.data"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# 2. Load the original dataset
columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

df = pd.read_csv(
    RAW_PATH,
    header=None,
    names=columns,
    na_values="?",
    skipinitialspace=True
)

df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=["target"]).copy()

# 3. Select four numerical features
features = ["age", "trestbps", "chol", "thalach"]
X = df[features]
y = (df["target"] > 0).astype(int)

# 4. Create one fixed train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Impute missing values using training data only
imputer = SimpleImputer(strategy="median")
X_train_clean = imputer.fit_transform(X_train)
X_test_clean = imputer.transform(X_test)

# 6. Encode features as rotation angles from 0 to pi
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train_encoded = scaler.fit_transform(X_train_clean)
X_test_encoded = scaler.transform(X_test_clean)

# 7. Train classical baseline on the same features
classical = LogisticRegression(max_iter=2000, random_state=42)

start = time.perf_counter()
classical.fit(X_train_encoded, y_train)
classical_time = time.perf_counter() - start

classical_pred = classical.predict(X_test_encoded)
classical_prob = classical.predict_proba(X_test_encoded)[:, 1]

# 8. Define the quantum feature map and trainable circuit
feature_map = z_feature_map(feature_dimension=4, reps=1)
ansatz = real_amplitudes(num_qubits=4, reps=1)

# Use fixed initial weights for reproducibility
rng = np.random.default_rng(42)
initial_point = rng.uniform(
    -np.pi,
    np.pi,
    size=ansatz.num_parameters
)

# 9. Create and train the Variational Quantum Classifier
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=100),
    initial_point=initial_point
)

start = time.perf_counter()
vqc.fit(X_train_encoded, y_train.to_numpy())
quantum_time = time.perf_counter() - start

quantum_pred = vqc.predict(X_test_encoded).astype(int).ravel()

# VQC's predict_proba returns class probabilities.
quantum_prob = vqc.predict_proba(X_test_encoded)[:, 1]

# 10. Evaluate both models
def evaluate(name, y_true, predictions, probabilities, seconds):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "Training_seconds": seconds
    }


results = [
    evaluate(
        "Logistic Regression (4 features)",
        y_test, classical_pred, classical_prob, classical_time
    ),
    evaluate(
        "Quantum VQC (4 qubits)",
        y_test, quantum_pred, quantum_prob, quantum_time
    )
]

results_df = pd.DataFrame(results)

print("\nCLASSICAL VS QUANTUM RESULTS")
print(results_df.round(3).to_string(index=False))

print("\nClassical confusion matrix:")
print(confusion_matrix(y_test, classical_pred, labels=[0, 1]))

print("\nQuantum confusion matrix:")
print(confusion_matrix(y_test, quantum_pred, labels=[0, 1]))

results_df.to_csv(ROOT / "quantum_comparison.csv", index=False)

# 11. Save models and preprocessing objects
joblib.dump(classical, MODELS / "logistic_regression_4features.joblib")
np.save(MODELS / "vqc_4qubits_weights.npy", vqc.weights)

print("Quantum model weights saved successfully.")
joblib.dump(imputer, MODELS / "quantum_imputer.joblib")
joblib.dump(scaler, MODELS / "quantum_scaler.joblib")

print("\nSaved comparison to quantum_comparison.csv")
print("Saved models and preprocessing objects.")
