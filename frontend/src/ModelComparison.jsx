
import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Cell,
} from "recharts";
import { BrainCircuit, Activity, Cpu, Info } from "lucide-react";
import "./ModelComparison.css";

// Previously reported results from your project.
// The quantum result uses four selected features.
// Logistic Regression and SVM below use the v2 preprocessing pipeline.
const models = [
  {
    name: "Logistic Regression",
    shortName: "Logistic Regression",
    accuracy: 86.9,
    description: "Classical baseline",
    icon: Activity,
  },
  {
    name: "SVM",
    shortName: "SVM",
    accuracy: 85.2,
    description: "Classical machine learning",
    icon: Cpu,
  },
  {
    name: "Quantum VQC",
    shortName: "Quantum VQC",
    accuracy: 45.9,
    description: "Hybrid quantum-classical",
    icon: BrainCircuit,
  },
];

const chartData = models.map((model) => ({
  model: model.shortName,
  accuracy: model.accuracy,
}));

function ModelComparison() {
  const highestAccuracy = models.reduce((best, current) =>
    current.accuracy > best.accuracy ? current : best
  );

  return (
    <section className="model-comparison" id="model-comparison">
      <div className="comparison-heading">
        <div>
          <span className="comparison-eyebrow">
            PERFORMANCE ANALYTICS
          </span>
          <h2>Model Comparison</h2>
          <p>
            Compare classical machine learning with a hybrid
            quantum-classical classifier.
          </p>
        </div>

        <div className="comparison-badge">
          <BrainCircuit size={18} />
          3 Models
        </div>
      </div>

      <div className="comparison-cards">
        {models.map((model) => {
          const Icon = model.icon;

          return (
            <article className="comparison-card" key={model.name}>
              <div className="comparison-card-top">
                <span className="comparison-icon">
                  <Icon size={22} />
                </span>

                {model.name === highestAccuracy.name && (
                  <span className="best-model-badge">
                    Highest reported accuracy
                  </span>
                )}
              </div>

              <h3>{model.name}</h3>
              <p className="model-description">
                {model.description}
              </p>

              <div className="accuracy-value">
                {model.accuracy.toFixed(1)}%
              </div>

              <div className="accuracy-track">
                <div
                  className={`accuracy-fill ${
                    model.name === "Quantum VQC"
                      ? "quantum-fill"
                      : ""
                  }`}
                  style={{ width: `${model.accuracy}%` }}
                />
              </div>

              <p className="accuracy-label">Reported accuracy</p>
            </article>
          );
        })}
      </div>

      <div className="comparison-chart-card">
        <div className="chart-heading">
          <div>
            <h3>Accuracy Comparison</h3>
            <p>Reported test accuracy by model</p>
          </div>
          <Activity size={22} />
        </div>

        <div className="comparison-chart">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              margin={{ top: 15, right: 15, left: 0, bottom: 10 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                vertical={false}
                stroke="#e5e7eb"
              />
              <XAxis
                dataKey="model"
                tick={{ fill: "#64748b", fontSize: 12 }}
                interval={0}
                angle={-10}
                textAnchor="end"
                height={65}
              />
              <YAxis
                domain={[0, 100]}
                tickFormatter={(value) => `${value}%`}
                tick={{ fill: "#64748b", fontSize: 12 }}
              />
              <Tooltip
                formatter={(value) => [`${value}%`, "Accuracy"]}
                contentStyle={{
                  borderRadius: "10px",
                  border: "1px solid #e2e8f0",
                }}
              />
              <Legend />
              <Bar
                dataKey="accuracy"
                name="Accuracy"
                radius={[6, 6, 0, 0]}
                maxBarSize={85}
              >
                {chartData.map((entry) => (
                  <Cell
                    key={entry.model}
                    fill={
                      entry.model === "Quantum VQC"
                        ? "#8b5cf6"
                        : entry.model === "SVM"
                        ? "#0ea5e9"
                        : "#2563eb"
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="comparison-insight">
        <div className="insight-icon">
          <Info size={21} />
        </div>
        <div>
          <h3>What do these results tell us?</h3>
          <p>
            Logistic Regression has the highest reported accuracy
            in these results. The quantum VQC result is lower, so
            this experiment does not demonstrate that the quantum
            model performs better than the classical models.
          </p>
          <p>
            <strong>Important evaluation note:</strong> These
            results come from different feature configurations.
            The classical v2 models use the full preprocessed
            feature set, while the quantum experiment uses four
            selected features. Retrain and evaluate all three
            models using the same train/test split and the same
            features before making a final performance claim.
          </p>
        </div>
      </div>

      <p className="comparison-disclaimer">
        Research prototype for educational purposes only. Model
        predictions are not a medical diagnosis.
      </p>
    </section>
  );
}

export default ModelComparison;
