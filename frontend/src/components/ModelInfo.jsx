export default function ModelInfo({ info, overall }) {
    return (
      <div className="model-info">
        <div className="model-stat">
          <span className="model-stat-val">{info.model_name?.split(" ")[0]}</span>
          <span className="model-stat-lbl">Model</span>
        </div>
        <div className="model-divider" />
        <div className="model-stat">
          <span className="model-stat-val">{Math.round((info.val_acc || 0) * 100)}%</span>
          <span className="model-stat-lbl">Val Acc</span>
        </div>
        <div className="model-divider" />
        <div className="model-stat">
          <span className="model-stat-val">
            {overall ? `${Math.round(overall * 100)}%` : "—"}
          </span>
          <span className="model-stat-lbl">Live Acc</span>
        </div>
        <div className="model-divider" />
        <div className="model-stat">
          <span className="model-stat-val">{info.version}</span>
          <span className="model-stat-lbl">Version</span>
        </div>
      </div>
    );
  }