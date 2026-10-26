
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "processed.cleveland.data"
OUT = ROOT / "data" / "processed_v2"
MODELS = ROOT / "models"

OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

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

# Convert the original 0-4 target into binary labels.
X = df.drop(columns="target")
y = (df["target"] > 0).astype(int)

numeric_features = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

categorical_features = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]

# Split raw records before learning preprocessing parameters.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Learn transformations on training records only.
X_train_ready = preprocessor.fit_transform(X_train)
X_test_ready = preprocessor.transform(X_test)

# Convert sparse outputs to dense arrays for CSV export.
if hasattr(X_train_ready, "toarray"):
    X_train_ready = X_train_ready.toarray()
    X_test_ready = X_test_ready.toarray()

feature_names = preprocessor.get_feature_names_out()

X_train_df = pd.DataFrame(
    X_train_ready, columns=feature_names
)
X_test_df = pd.DataFrame(
    X_test_ready, columns=feature_names
)

X_train_df.to_csv(OUT / "X_train.csv", index=False)
X_test_df.to_csv(OUT / "X_test.csv", index=False)
y_train.to_csv(OUT / "y_train.csv", index=False)
y_test.to_csv(OUT / "y_test.csv", index=False)

joblib.dump(preprocessor, MODELS / "preprocessor_v2.joblib")

print("Improved preprocessing completed.")
print("Training shape:", X_train_df.shape)
print("Testing shape:", X_test_df.shape)
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)
print("Feature names:")
print(feature_names.tolist())
print("Saved to:", OUT)
