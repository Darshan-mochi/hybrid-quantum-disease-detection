import { useState } from "react";
import "./App.css";

import ExperimentDashboard from "./ExperimentDashboard";
import ModelComparison from "./ModelComparison";

const initialValues = {
  age: 55,
  sex: 1,
  cp: 4,
  trestbps: 140,
  chol: 250,
  fbs: 0,
  restecg: 1,
  thalach: 150,
  exang: 0,
  oldpeak: 1.5,
  slope: 2,
  ca: 0,
  thal: 3,
};

const fields = [
  ["age", "Age", 1, 120, 1],
  ["sex", "Sex (0 or 1)", 0, 1, 1],
  ["cp", "Chest pain type (1–4)", 1, 4, 1],
  ["trestbps", "Resting blood pressure", 1, 300, 1],
  ["chol", "Cholesterol", 0, 1000, 1],
  ["fbs", "Fasting blood sugar (0 or 1)", 0, 1, 1],
  ["restecg", "Resting ECG (0–2)", 0, 2, 1],
  ["thalach", "Maximum heart rate", 1, 300, 1],
  ["exang", "Exercise-induced angina (0 or 1)", 0, 1, 1],
  ["oldpeak", "Oldpeak", -10, 20, 0.1],
  ["slope", "Slope (1–3)", 1, 3, 1],
  ["ca", "Major vessels (0–3)", 0, 3, 1],
  ["thal", "Thal dataset code (3–7)", 3, 7, 1],
];

export default function App() {
  // Keep the existing experiment dashboard route.
  if (window.location.pathname === "/experiments") {
    return <ExperimentDashboard />;
  }

  const [values, setValues] = useState(initialValues);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function updateField(name, value) {
    setValues((previous) => ({
      ...previous,
      [name]: value,
    }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setResult(null);
    setError("");

    try {
      const API_URL = import.meta.env.API_URL;
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(
          Object.fromEntries(
            Object.entries(values).map(([key, value]) => [key, Number(value)]),
          ),
        ),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Prediction request failed",
        );
      }

      setResult(data);
    } catch (err) {
      setError(`${err.message}. Check that the FastAPI server is running.`);
    } finally {
      setLoading(false);
    }
  }

  const isPositive = result?.label === "positive_class";

  return (
    <main className="page">
      <header className="header">
        <div>
          <p className="eyebrow">SIH · MACHINE LEARNING RESEARCH</p>
          <h1>Hybrid Quantum Disease Detection</h1>
          <p className="subtitle">
            Heart-disease classification research prototype
          </p>
        </div>

        <div className="header-actions">
          <span className="status">Local demo</span>
          <a className="nav-link" href="/experiments">
            Experiment dashboard
          </a>
        </div>
      </header>

      <section className="notice">
        <strong>Research use only</strong>
        <p>
          This educational prototype is not clinically validated. Its output
          must not be used to diagnose disease or guide treatment.
        </p>
      </section>

      <section className="panel">
        <h2>Input features</h2>
        <p className="muted">
          Enter the numerical values used by the training dataset.
        </p>

        <form onSubmit={handleSubmit}>
          <div className="fields">
            {fields.map(([name, label, min, max, step]) => (
              <label className="field" key={name}>
                <span>{label}</span>
                <input
                  type="number"
                  min={min}
                  max={max}
                  step={step}
                  value={values[name]}
                  onChange={(event) => updateField(name, event.target.value)}
                  required
                />
              </label>
            ))}
          </div>

          <button type="submit" disabled={loading}>
            {loading ? "Running model..." : "Run research prediction"}
          </button>
        </form>
      </section>

      {error && (
        <section className="panel error" role="alert">
          <h2>Request failed</h2>
          <p>{error}</p>
        </section>
      )}

      {result && (
        <section
          className={`panel result ${isPositive ? "" : "negative"}`}
          aria-live="polite"
        >
          <p className="eyebrow">MODEL OUTPUT</p>

          <h2>
            {isPositive
              ? "Positive class predicted"
              : "Negative class predicted"}
          </h2>

          <p>
            Estimated positive-class probability:{" "}
            <strong>
              {(result.positive_class_probability * 100).toFixed(2)}%
            </strong>
          </p>

          <div className="progress">
            <div
              className="progressFill"
              style={{
                width: `${result.positive_class_probability * 100}%`,
              }}
            />
          </div>

          <p className="muted">{result.notice}</p>
        </section>
      )}

      {/* Model comparison appears at the end of the main page. */}
      <ModelComparison />

      <footer>
        Research prototype · UCI Heart Disease dataset · Not for clinical use
      </footer>
    </main>
  );
}
