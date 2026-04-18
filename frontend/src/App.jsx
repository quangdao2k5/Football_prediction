import { useState, useEffect } from "react";
import Predictions from "./components/Predictions";
import Accuracy from "./components/Accuracy";
import Standings from "./components/Standings";
import ModelInfo from "./components/ModelInfo";

const API = "http://localhost:8000/api";

export default function App() {
  const [tab, setTab]           = useState("predictions");
  const [predictions, setPred]  = useState(null);
  const [accuracy, setAcc]      = useState(null);
  const [standings, setStand]   = useState(null);
  const [modelInfo, setModel]   = useState(null);
  const [loading, setLoading]   = useState(true);

  useEffect(() => {
    async function fetchAll() {
      setLoading(true);
      try {
        const [pred, acc, stand, model] = await Promise.all([
          fetch(`${API}/predictions/latest`).then(r => r.ok ? r.json() : null).catch(() => null),
          fetch(`${API}/accuracy`).then(r => r.ok ? r.json() : null).catch(() => null),
          fetch(`${API}/standings`).then(r => r.ok ? r.json() : null).catch(() => null),
          fetch(`${API}/model/info`).then(r => r.ok ? r.json() : null).catch(() => null),
        ]);
        setPred(pred);
        setAcc(acc);
        setStand(stand);
        setModel(model);
      } catch (e) {
        console.error("Loi ket noi API:", e);
      }
      setLoading(false);
    }
    fetchAll();
  }, []);

  const tabs = [
    { id: "predictions", label: "Dự đoán" },
    { id: "accuracy",    label: "Độ chính xác" },
    { id: "standings",   label: "Bảng xếp hạng" },
  ];

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-badge">EPL</span>
            <div>
              <h1>Match Predictor</h1>
              <p>English Premier League 2025/26</p>
            </div>
          </div>
          {modelInfo && (
            <ModelInfo info={modelInfo} overall={accuracy?.overall} />
          )}
        </div>

        <nav className="tabs">
          {tabs.map(t => (
            <button
              key={t.id}
              className={`tab-btn ${tab === t.id ? "active" : ""}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="main">
        {loading ? (
          <div className="loading">
            <div className="spinner" />
            <p>Đang tải dữ liệu...</p>
          </div>
        ) : (
          <>
            {tab === "predictions" && <Predictions data={predictions} />}
            {tab === "accuracy"    && <Accuracy data={accuracy} />}
            {tab === "standings"   && <Standings data={standings} />}
          </>
        )}
      </main>
    </div>
  );
}