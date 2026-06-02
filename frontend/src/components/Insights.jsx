const API_HOST = "http://localhost:8000";

export default function Insights({ modelInfo, accuracy }) {
  if (!modelInfo) {
    return <div className="empty">Chưa có thông tin model.</div>;
  }

  const pct = (value, digits = 1) => {
    if (value == null) return "—";
    return `${(value * 100).toFixed(digits)}%`;
  };

  const reportSrc = (path) => {
    if (!path) return null;
    return path.startsWith("http") ? path : `${API_HOST}${path}`;
  };

  const cmSrc = reportSrc(modelInfo.reports?.confusion_matrix);
  const fiSrc = reportSrc(modelInfo.reports?.feature_importance);

  return (
    <div className="section">
      <div className="section-header">
        <div>
          <h2>Phân tích mô hình hiện tại</h2>
          <p className="section-subtitle">Các biểu đồ này được tạo từ model đang dùng để dự đoán trên giao diện.</p>
        </div>
        <span className="badge">{modelInfo.version}</span>
      </div>

      <div className="model-summary-panel">
        <div>
          <div className="summary-eyebrow">Model đang sử dụng</div>
          <div className="summary-title">{modelInfo.model_name}</div>
          <div className="summary-meta">
            {modelInfo.feature_count || 0} features · draw boost {Number(modelInfo.draw_boost || 0).toFixed(2)}
          </div>
        </div>

        <div className="summary-metrics">
          <div className="metric-tile">
            <span className="metric-value">{pct(modelInfo.val_acc)}</span>
            <span className="metric-label">Validation</span>
          </div>
          <div className="metric-tile">
            <span className="metric-value">{pct(modelInfo.test_acc)}</span>
            <span className="metric-label">Test season</span>
          </div>
          <div className="metric-tile">
            <span className="metric-value">{pct(accuracy?.overall)}</span>
            <span className="metric-label">GW31-38</span>
          </div>
          <div className="metric-tile">
            <span className="metric-value">
              {accuracy?.total_correct ?? "—"}/{accuracy?.total_matches ?? "—"}
            </span>
            <span className="metric-label">Trận đã đánh giá</span>
          </div>
        </div>
      </div>

      <div className="report-grid">
        <div className="report-panel">
          <div className="report-header">
            <h3>Ma trận nhầm lẫn</h3>
            <span>Dự đoán H/D/A trên tập test</span>
          </div>
          {cmSrc ? (
            <img src={`${cmSrc}?v=${modelInfo.version}`} alt="Confusion matrix của model hiện tại" />
          ) : (
            <div className="empty compact">Chưa có confusion matrix.</div>
          )}
        </div>

        <div className="report-panel">
          <div className="report-header">
            <h3>Feature importance</h3>
            <span>Các biến ảnh hưởng mạnh nhất</span>
          </div>
          {fiSrc ? (
            <img src={`${fiSrc}?v=${modelInfo.version}`} alt="Feature importance của model hiện tại" />
          ) : (
            <div className="empty compact">Chưa có feature importance.</div>
          )}
        </div>
      </div>
    </div>
  );
}
