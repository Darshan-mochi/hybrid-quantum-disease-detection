import { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

const API = import.meta.env.API_URL;

const tooltipStyle = {
  borderRadius: 14,
  border: "1px solid #c4b5fd",
  background: "#fff",
  padding: 12,
  fontFamily: '"Plus Jakarta Sans", sans-serif',
};

export default function ExperimentDashboard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API}/experiments`)
      .then(async (response) => {
        if (!response.ok) {
          const body = await response.json();
          throw new Error(body.detail || "Unable to load experiments");
        }
        return response.json();
      })
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const pageHeader = (
    <header className="header">
      <div>
        <p className="eyebrow">SIH · MODEL EVALUATION</p>
        <h1>Experiment Dashboard</h1>
        <p className="subtitle">Recorded classical and quantum model results</p>
      </div>
      <a className="nav-link" href="/">
        Prediction page
      </a>
    </header>
  );

  if (loading) {
    return (
      <main className="page">
        {pageHeader}
        <section className="panel state-card">Loading experiments...</section>
      </main>
    );
  }

  if (error) {
    return (
      <main className="page">
        {pageHeader}
        <section className="panel error state-card">
          Could not load results: {error}
        </section>
      </main>
    );
  }

  if (!data) return null;

  const baseline = data.classical_baseline;

  const rows = [
    {
      experiment: "Logistic Regression (4-feature baseline)",
      accuracy: baseline.accuracy * 100,
      recall: baseline.recall * 100,
      training_seconds: baseline.training_seconds,
    },
    ...data.experiments.map((item) => ({
      experiment: item.experiment,
      accuracy: item.accuracy * 100,
      recall: item.recall * 100,
      training_seconds: item.training_seconds,
    })),
  ];

  const featureRows = data.feature_importance.map((item) => ({
    feature: item.feature,
    importance: item.importance_mean,
  }));

  return (
    <main className="page">
      {pageHeader}

      <section className="notice">
        <strong>Research results only</strong>
        <p>{data.notice}</p>
      </section>

      <section className="panel">
        <h2>Accuracy and recall</h2>
        <p className="muted">Percentages from your recorded experiments.</p>
        <div className="chart-wrap">
          <ResponsiveContainer>
            <BarChart data={rows} margin={{ bottom: 65, left: 8, right: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ddd6fe" />
              <XAxis
                dataKey="experiment"
                angle={-18}
                textAnchor="end"
                interval={0}
                height={90}
                tick={{ fontSize: 11, fill: "#334155" }}
              />
              <YAxis domain={[0, 100]} tick={{ fill: "#334155" }} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend />
              <Bar
                dataKey="accuracy"
                name="Accuracy (%)"
                fill="#7c3aed"
                radius={[8, 8, 0, 0]}
              />
              <Bar
                dataKey="recall"
                name="Recall (%)"
                fill="#06b6d4"
                radius={[8, 8, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="panel">
        <h2>Training time</h2>
        <div className="chart-wrap short">
          <ResponsiveContainer>
            <BarChart data={rows} margin={{ bottom: 55, left: 8, right: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#fecdd3" />
              <XAxis
                dataKey="experiment"
                tick={{ fontSize: 10, fill: "#334155" }}
                angle={-15}
                textAnchor="end"
                height={75}
                interval={0}
              />
              <YAxis tick={{ fill: "#334155" }} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar
                dataKey="training_seconds"
                name="Training time (seconds)"
                fill="#f43f5e"
                radius={[8, 8, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="panel">
        <h2>Feature importance</h2>
        <p className="muted">
          Mean accuracy decrease after shuffling each encoded feature.
        </p>
        {featureRows.length === 0 ? (
          <p>No feature-importance data found.</p>
        ) : (
          <div className="chart-wrap">
            <ResponsiveContainer>
              <BarChart
                data={featureRows}
                layout="vertical"
                margin={{ left: 35, right: 25, top: 8, bottom: 8 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#bbf7d0" />
                <XAxis type="number" tick={{ fill: "#334155" }} />
                <YAxis
                  type="category"
                  dataKey="feature"
                  width={160}
                  tick={{ fontSize: 11, fill: "#334155" }}
                />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar
                  dataKey="importance"
                  name="Mean importance"
                  fill="#22c55e"
                  radius={[0, 8, 8, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>

      <p
        className="muted"
        style={{ color: "rgba(255,255,255,0.78)", padding: "8px 8px 0" }}
      >
        The classical baseline uses four features. The recorded quantum
        experiments also use four features. Compare these only as research runs;
        repeated trials and uncertainty estimates are needed for stronger
        conclusions.
      </p>
    </main>
  );
}
