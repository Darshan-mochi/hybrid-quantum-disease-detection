
from pathlib import Path

import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 1. Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed.cleveland.data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# 2. Define the dataset column names
columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]


# 3. Load the raw dataset
df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=columns,
    na_values="?",
    skipinitialspace=True
)

# Convert all columns to numeric values
df = df.apply(pd.to_numeric, errors="coerce")

print("Original dataset shape:", df.shape)
print("\nMissing values before cleaning:")
print(df.isnull().sum())


# 4. Remove records without a known target
df = df.dropna(subset=["target"]).copy()

# 5. Encode the target for binary classification
y = (df["target"] > 0).astype(int)

# 6. Separate features from the target
X = df.drop(columns=["target"])

print("\nBinary target distribution:")
print(y.value_counts().sort_index())


# 7. Split before fitting preprocessing steps
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 8. Impute missing feature values
imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)


# 9. Scale features using training data only
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)


# 10. Save prepared datasets as DataFrames
feature_names = X.columns.tolist()

X_train_df = pd.DataFrame(
    X_train_scaled, columns=feature_names
)
X_test_df = pd.DataFrame(
    X_test_scaled, columns=feature_names
)

X_train_df.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
X_test_df.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)


# 11. Save preprocessing objects for future predictions
joblib.dump(imputer, MODEL_DIR / "imputer.joblib")
joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

print("\nPreprocessing completed successfully!")
print("Training features:", X_train_df.shape)
print("Testing features:", X_test_df.shape)
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)
print("\nSaved processed files to:", OUTPUT_DIR)
