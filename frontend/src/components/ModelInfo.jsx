export default function ModelInfo({ info, overall }) {
    const pct = (v) => v == null ? "—" : `${Math.round(v * 100)}%`;
    const modelName = info.model_name || "Model";

    return (
      <div className="model-info">
        <div className="model-stat">
          <span className="model-stat-val model-name" title={modelName}>{modelName}</span>
          <span className="model-stat-lbl">Model</span>
        </div>
        <div className="model-divider" />
        <div className="model-stat">
          <span className="model-stat-val">{pct(info.val_acc)}</span>
          <span className="model-stat-lbl">Val Acc</span>
        </div>
        <div className="model-divider" />
        <div className="model-stat">
          <span className="model-stat-val">{pct(info.test_acc)}</span>
          <span className="model-stat-lbl">Test Acc</span>
        </div>
        <div className="model-divider" />
        <div className="model-stat">
          <span className="model-stat-val">{pct(overall)}</span>
          <span className="model-stat-lbl">GW Acc</span>
        </div>
        <div className="model-divider" />
        <div className="model-stat">
          <span className="model-stat-val">{info.version}</span>
          <span className="model-stat-lbl">Version</span>
        </div>
      </div>
    );
  }
