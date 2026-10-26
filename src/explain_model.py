
import pandas as pd
import joblib

from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed_v2"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "data" / "experiments"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the preprocessed data
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

# Load the trained classical model
model = joblib.load(
    MODELS_DIR / "logistic_regression_v2.joblib"
)

# Calculate baseline performance
baseline_accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
)

print(f"Baseline accuracy: {baseline_accuracy:.3f}")

# Shuffle each feature and measure the performance change
result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="accuracy",
    n_repeats=20,
    random_state=42,
    n_jobs=1
)

importance = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
})

importance = importance.sort_values(
    "importance_mean",
    ascending=False
)

output_file = OUTPUT_DIR / "feature_importance.csv"
importance.to_csv(output_file, index=False)

print("\nFEATURE IMPORTANCE")
print(importance.to_string(index=False))

print(f"\nSaved results to: {output_file}")

