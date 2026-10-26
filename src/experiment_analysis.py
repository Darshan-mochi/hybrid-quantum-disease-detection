
import pandas as pd
from pathlib import Path

# Project root directory
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Results reported from your three experiments
experiments = [
    {
        "experiment": "VQC_baseline",
        "iterations": 20,
        "fixed_initial_weights": False,
        "accuracy": 0.475,
        "recall": 0.357,
        "training_seconds": 19.519,
    },
    {
        "experiment": "VQC_100_iterations",
        "iterations": 100,
        "fixed_initial_weights": False,
        "accuracy": 0.443,
        "recall": 0.214,
        "training_seconds": 49.523,
    },
    {
        "experiment": "VQC_fixed_weights",
        "iterations": 100,
        "fixed_initial_weights": True,
        "accuracy": 0.459,
        "recall": 0.214,
        "training_seconds": 57.146,
    },
]

df = pd.DataFrame(experiments)

# Convert metrics to percentages for display
df["accuracy_percent"] = (df["accuracy"] * 100).round(1)
df["recall_percent"] = (df["recall"] * 100).round(1)

# Save experiment history
output_file = RESULTS_DIR / "quantum_experiments.csv"
df.to_csv(output_file, index=False)

print("\nQUANTUM EXPERIMENT COMPARISON")
print(
    df[
        [
            "experiment",
            "accuracy_percent",
            "recall_percent",
            "training_seconds",
        ]
    ].to_string(index=False)
)

print(f"\nResults saved to: {output_file}")

# Identify the best recorded accuracy
best = df.loc[df["accuracy"].idxmax()]
print(
    f"\nBest recorded accuracy: {best['experiment']} "
    f"({best['accuracy_percent']}%)"
)
