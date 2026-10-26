
from pathlib import Path

import joblib
import pandas as pd


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed_v2"

MODEL_PATH = MODELS_DIR / "logistic_regression_v2.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor_v2.joblib"
TRAIN_COLUMNS_PATH = DATA_DIR / "X_train.csv"

app = FastAPI(
    title="Hybrid Quantum Disease Detection API",
    description="Research prototype for heart-disease classification.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Load the trained classifier and its original preprocessing transformer.
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# These are the transformed columns used when training the model.
model_columns = pd.read_csv(
    TRAIN_COLUMNS_PATH, nrows=0
).columns.tolist()

RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal",
]


class PatientFeatures(BaseModel):
    age: float = Field(ge=1, le=120)
    sex: float = Field(ge=0, le=1)
    cp: float = Field(ge=1, le=4)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(ge=0, le=1000)
    fbs: float = Field(ge=0, le=1)
    restecg: float = Field(ge=0, le=2)
    thalach: float = Field(gt=0, le=300)
    exang: float = Field(ge=0, le=1)
    oldpeak: float = Field(ge=-10, le=20)
    slope: float = Field(ge=1, le=3)
    ca: float = Field(ge=0, le=3)
    thal: float = Field(ge=3, le=7)


@app.get("/")
def home():
    return {
        "message": "Heart-disease research API is running.",
        "docs": "/docs",
    }


@app.post("/predict")
def predict(patient: PatientFeatures):
    try:
        # Step 1: Receive the original 13 input features.
        raw_df = pd.DataFrame(
            [patient.model_dump()],
            columns=RAW_COLUMNS,
        )

        # Step 2: Apply exactly the preprocessing used in training.
        transformed = preprocessor.transform(raw_df)

        # Step 3: Restore the transformed feature names expected by
        # the classifier.
        transformed_df = pd.DataFrame(
            transformed,
            columns=model_columns,
        )

        # Step 4: Predict using the trained classifier.
        prediction = int(model.predict(transformed_df)[0])
        probabilities = model.predict_proba(transformed_df)
        probability = float(probabilities[0][1])

        return {
            "prediction": prediction,
            "label": (
                "positive_class"
                if prediction == 1
                else "negative_class"
            ),
            "positive_class_probability": round(probability, 4),
            "notice": (
                "Research prototype only; not a medical diagnosis."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}: {exc}",
        )


@app.get("/experiments")
def get_experiments():
    results_path = (
        ROOT / "data" / "experiments" / "quantum_experiments.csv"
    )
    importance_path = (
        ROOT / "data" / "experiments" / "feature_importance.csv"
    )

    if not results_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Experiment results file not found.",
        )

    experiments_df = pd.read_csv(results_path)
    importance = []

    if importance_path.exists():
        importance_df = pd.read_csv(importance_path)
        importance = importance_df.head(10).to_dict(
            orient="records"
        )

    # Classical baseline values from your previously reported run.
    # These use four features and are not the full 13-feature model.
    classical_baseline = {
        "experiment": "Logistic Regression (4-feature baseline)",
        "accuracy": 0.754,
        "recall": 0.679,
        "training_seconds": 0.011,
    }

    return {
        "experiments": experiments_df.to_dict(orient="records"),
        "classical_baseline": classical_baseline,
        "feature_importance": importance,
        "notice": (
            "Research results only. Runs may differ due to initialization "
            "and configuration. Not for clinical use."
        ),
    }
