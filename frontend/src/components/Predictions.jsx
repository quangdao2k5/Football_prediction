export default function Predictions({ data }) {
    if (!data) return <div className="empty">Chưa có dữ liệu dự đoán.</div>;
  
    const { gameweek, matches } = data;
  
    const resultColor = (pred) => {
      if (pred === "Home Win") return "var(--green)";
      if (pred === "Away Win") return "var(--red)";
      return "var(--yellow)";
    };
  
    return (
      <div className="section">
        <div className="section-header">
          <h2>Dự đoán Gameweek {gameweek}</h2>
          <span className="badge">{matches.length} trận</span>
        </div>
  
        <div className="match-grid">
          {matches.map((m, i) => {
            const maxProb = Math.max(m["home_win%"], m["draw%"], m["away_win%"]);
            return (
              <div className="match-card" key={i}>
                <div className="match-date">{m.date}</div>
  
                <div className="match-teams">
                  <span className={`team ${m.prediction === "Home Win" ? "winner" : ""}`}>
                    {m.home}
                  </span>
                  <span className="vs">vs</span>
                  <span className={`team right ${m.prediction === "Away Win" ? "winner" : ""}`}>
                    {m.away}
                  </span>
                </div>
  
                <div className="prob-bars">
                  <div className="prob-bar-row">
                    <span className="prob-label">H</span>
                    <div className="prob-bar-bg">
                      <div
                        className="prob-bar-fill home"
                        style={{ width: `${m["home_win%"]}%` }}
                      />
                    </div>
                    <span className="prob-val">{m["home_win%"]}%</span>
                  </div>
                  <div className="prob-bar-row">
                    <span className="prob-label">D</span>
                    <div className="prob-bar-bg">
                      <div
                        className="prob-bar-fill draw"
                        style={{ width: `${m["draw%"]}%` }}
                      />
                    </div>
                    <span className="prob-val">{m["draw%"]}%</span>
                  </div>
                  <div className="prob-bar-row">
                    <span className="prob-label">A</span>
                    <div className="prob-bar-bg">
                      <div
                        className="prob-bar-fill away"
                        style={{ width: `${m["away_win%"]}%` }}
                      />
                    </div>
                    <span className="prob-val">{m["away_win%"]}%</span>
                  </div>
                </div>
  
                <div
                  className="prediction-label"
                  style={{ color: resultColor(m.prediction) }}
                >
                  {m.prediction}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }