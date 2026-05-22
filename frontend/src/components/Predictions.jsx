import { useState } from "react";

export default function Predictions({ data }) {
    const [selectedMatch, setSelectedMatch] = useState(null);

    if (!data) return <div className="empty">Chưa có dữ liệu dự đoán.</div>;
  
    const { gameweek, matches } = data;
  
    const resultColor = (pred) => {
      if (pred === "Home Win") return "var(--green)";
      if (pred === "Away Win") return "var(--red)";
      return "var(--yellow)";
    };

    const resultLabel = (pred) => {
      if (pred === "Home Win") return "Đội nhà thắng";
      if (pred === "Away Win") return "Đội khách thắng";
      return "Hoà";
    };

    const renderForm = (formStr) => {
      if (!formStr || formStr === "N/A") return <span className="stat-val">N/A</span>;
      return (
        <div className="form-badges">
          {formStr.split(" ").map((res, i) => (
            <span key={i} className={`form-badge ${res}`}>{res}</span>
          ))}
        </div>
      );
    };

    const getOvr = (elo) => {
      if (!elo) return 50;
      return Math.min(99, Math.max(50, Math.round(50 + (elo - 1200) / 12)));
    };

    const getConfidence = (m) => {
      const maxProb = Math.max(m["home_win%"], m["draw%"], m["away_win%"]);
      if (maxProb >= 70) return { level: "Rất cao", color: "var(--green)", icon: "🔥" };
      if (maxProb >= 55) return { level: "Cao", color: "var(--accent)", icon: "💪" };
      if (maxProb >= 45) return { level: "Trung bình", color: "var(--yellow)", icon: "⚖️" };
      return { level: "Thấp", color: "var(--text-muted)", icon: "🤔" };
    };

    const getFormPoints = (formStr) => {
      if (!formStr || formStr === "N/A") return 0;
      let pts = 0;
      formStr.split(" ").forEach(r => {
        if (r === "W") pts += 3;
        else if (r === "D") pts += 1;
      });
      return pts;
    };

    // SVG donut chart for probabilities
    const ProbDonut = ({ homeP, drawP, awayP, prediction }) => {
      const r = 54, cx = 64, cy = 64, strokeWidth = 14;
      const circ = 2 * Math.PI * r;
      const gap = 4; // gap in px between segments
      const totalGap = gap * 3;
      const usable = circ - totalGap;

      const homeLen = (homeP / 100) * usable;
      const drawLen = (drawP / 100) * usable;
      const awayLen = (awayP / 100) * usable;

      const homeOffset = 0;
      const drawOffset = homeLen + gap;
      const awayOffset = homeLen + gap + drawLen + gap;

      const maxProb = Math.max(homeP, drawP, awayP);
      const centerText = `${maxProb}%`;

      return (
        <svg width="128" height="128" viewBox="0 0 128 128" className="prob-donut">
          {/* Background circle */}
          <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--border)" strokeWidth={strokeWidth} opacity="0.3" />
          {/* Home segment */}
          <circle cx={cx} cy={cy} r={r} fill="none"
            stroke="var(--green)" strokeWidth={strokeWidth}
            strokeDasharray={`${homeLen} ${circ - homeLen}`}
            strokeDashoffset={-homeOffset}
            strokeLinecap="round"
            transform={`rotate(-90 ${cx} ${cy})`}
            style={{ transition: "stroke-dasharray 0.8s ease" }}
          />
          {/* Draw segment */}
          <circle cx={cx} cy={cy} r={r} fill="none"
            stroke="var(--yellow)" strokeWidth={strokeWidth}
            strokeDasharray={`${drawLen} ${circ - drawLen}`}
            strokeDashoffset={-drawOffset}
            strokeLinecap="round"
            transform={`rotate(-90 ${cx} ${cy})`}
            style={{ transition: "stroke-dasharray 0.8s ease" }}
          />
          {/* Away segment */}
          <circle cx={cx} cy={cy} r={r} fill="none"
            stroke="var(--red)" strokeWidth={strokeWidth}
            strokeDasharray={`${awayLen} ${circ - awayLen}`}
            strokeDashoffset={-awayOffset}
            strokeLinecap="round"
            transform={`rotate(-90 ${cx} ${cy})`}
            style={{ transition: "stroke-dasharray 0.8s ease" }}
          />
          {/* Center text */}
          <text x={cx} y={cy - 4} textAnchor="middle" dominantBaseline="central"
            fill="var(--text)" fontSize="22" fontWeight="700" fontFamily="Barlow Condensed, sans-serif">
            {centerText}
          </text>
          <text x={cx} y={cy + 16} textAnchor="middle" dominantBaseline="central"
            fill="var(--text-muted)" fontSize="9" fontFamily="DM Sans, sans-serif" textTransform="uppercase">
            {prediction === "Home Win" ? "HOME" : prediction === "Away Win" ? "AWAY" : "DRAW"}
          </text>
        </svg>
      );
    };

    // Stat comparison bar
    const ComparisonBar = ({ homeVal, awayVal, label, suffix = "", higherIsBetter = true, isPercentage = false }) => {
      const hv = parseFloat(homeVal) || 0;
      const av = parseFloat(awayVal) || 0;
      const maxVal = Math.max(hv, av, 0.1);
      const homeWidth = (hv / (hv + av || 1)) * 100;
      const awayWidth = 100 - homeWidth;

      const homeWins = higherIsBetter ? hv > av : hv < av;
      const awayWins = higherIsBetter ? av > hv : av < hv;

      const formatVal = (v) => {
        if (isPercentage) return `${Math.round(v * 100)}%`;
        return `${v}${suffix}`;
      };

      return (
        <div className="detail-stat-row">
          <div className={`detail-stat-value left ${homeWins ? 'highlight' : ''}`}>
            {formatVal(hv)}
          </div>
          <div className="detail-stat-center">
            <div className="detail-stat-bar">
              <div className="detail-bar-fill home" style={{ width: `${homeWidth}%` }} />
              <div className="detail-bar-fill away" style={{ width: `${awayWidth}%` }} />
            </div>
            <div className="detail-stat-label">{label}</div>
          </div>
          <div className={`detail-stat-value right ${awayWins ? 'highlight' : ''}`}>
            {formatVal(av)}
          </div>
        </div>
      );
    };

    // OVR circle
    const OvrCircle = ({ ovr, side }) => {
      const r = 28, cx = 32, cy = 32, strokeWidth = 5;
      const circ = 2 * Math.PI * r;
      const fill = (ovr / 100) * circ;
      const color = ovr >= 85 ? "var(--green)" : ovr >= 75 ? "var(--accent)" : ovr >= 65 ? "var(--yellow)" : "var(--red)";
      return (
        <div className={`ovr-circle ${side}`}>
          <svg width="64" height="64" viewBox="0 0 64 64">
            <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--border)" strokeWidth={strokeWidth} opacity="0.3" />
            <circle cx={cx} cy={cy} r={r} fill="none"
              stroke={color} strokeWidth={strokeWidth}
              strokeDasharray={`${fill} ${circ - fill}`}
              strokeLinecap="round"
              transform={`rotate(-90 ${cx} ${cy})`}
              style={{ transition: "stroke-dasharray 0.6s ease" }}
            />
            <text x={cx} y={cy} textAnchor="middle" dominantBaseline="central"
              fill={color} fontSize="16" fontWeight="700" fontFamily="Barlow Condensed, sans-serif">
              {ovr}
            </text>
          </svg>
          <span className="ovr-label">OVR</span>
        </div>
      );
    };

    return (
      <div className="section">
        <div className="section-header">
          <h2>Dự đoán Gameweek {gameweek}</h2>
          <span className="badge">{matches.length} trận</span>
        </div>
  
        <div className="match-grid">
          {matches.map((m, i) => {
            return (
              <div className="match-card clickable" key={i} onClick={() => setSelectedMatch(m)}>
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
                      <div className="prob-bar-fill home" style={{ width: `${m["home_win%"]}%` }} />
                    </div>
                    <span className="prob-val">{m["home_win%"]}%</span>
                  </div>
                  <div className="prob-bar-row">
                    <span className="prob-label">D</span>
                    <div className="prob-bar-bg">
                      <div className="prob-bar-fill draw" style={{ width: `${m["draw%"]}%` }} />
                    </div>
                    <span className="prob-val">{m["draw%"]}%</span>
                  </div>
                  <div className="prob-bar-row">
                    <span className="prob-label">A</span>
                    <div className="prob-bar-bg">
                      <div className="prob-bar-fill away" style={{ width: `${m["away_win%"]}%` }} />
                    </div>
                    <span className="prob-val">{m["away_win%"]}%</span>
                  </div>
                </div>
  
                <div className="prediction-label" style={{ color: resultColor(m.prediction) }}>
                  {m.prediction}
                </div>
              </div>
            );
          })}
        </div>

        {/* ====== REDESIGNED MODAL ====== */}
        {selectedMatch && (() => {
          const m = selectedMatch;
          const homeOvr = getOvr(m.home_elo);
          const awayOvr = getOvr(m.away_elo);
          const confidence = getConfidence(m);
          const homeFormPts = getFormPoints(m.home_form_str);
          const awayFormPts = getFormPoints(m.away_form_str);

          // Parse H2H string
          const h2hMatch = (m.h2h_str || "").match(/Thắng (\d+).*Hòa (\d+).*Thua (\d+)/);
          const h2hWins = h2hMatch ? parseInt(h2hMatch[1]) : 0;
          const h2hDraws = h2hMatch ? parseInt(h2hMatch[2]) : 0;
          const h2hLosses = h2hMatch ? parseInt(h2hMatch[3]) : 0;
          const h2hTotal = h2hWins + h2hDraws + h2hLosses || 1;

          return (
            <div className="modal-overlay" onClick={() => setSelectedMatch(null)}>
              <div className="detail-modal" onClick={(e) => e.stopPropagation()}>
                <button className="close-btn" onClick={() => setSelectedMatch(null)}>✕</button>
                
                {/* ── Header with teams and OVR ── */}
                <div className="detail-header">
                  <div className="detail-team-block home">
                    <OvrCircle ovr={homeOvr} side="home" />
                    <div className="detail-team-name">{m.home}</div>
                    <div className="detail-team-sub">Đội nhà</div>
                  </div>
                  
                  <div className="detail-vs-block">
                    <div className="detail-date">{m.date}</div>
                    <div className="detail-vs-text">VS</div>
                    <div className="detail-gw">Vòng {m.gameweek}</div>
                  </div>

                  <div className="detail-team-block away">
                    <OvrCircle ovr={awayOvr} side="away" />
                    <div className="detail-team-name">{m.away}</div>
                    <div className="detail-team-sub">Đội khách</div>
                  </div>
                </div>

                {/* ── Prediction Result ── */}
                <div className="detail-prediction-section">
                  <div className="detail-donut-area">
                    <ProbDonut 
                      homeP={m["home_win%"]} 
                      drawP={m["draw%"]} 
                      awayP={m["away_win%"]} 
                      prediction={m.prediction} 
                    />
                  </div>
                  <div className="detail-prediction-info">
                    <div className="detail-pred-label">Dự đoán AI</div>
                    <div className="detail-pred-result" style={{ color: resultColor(m.prediction) }}>
                      {resultLabel(m.prediction)}
                    </div>
                    <div className="detail-confidence">
                      <span className="confidence-icon">{confidence.icon}</span>
                      <span>Độ tin cậy: </span>
                      <strong style={{ color: confidence.color }}>{confidence.level}</strong>
                    </div>
                    <div className="detail-prob-legend">
                      <span className="prob-legend-item"><span className="legend-dot home"></span>Nhà {m["home_win%"]}%</span>
                      <span className="prob-legend-item"><span className="legend-dot draw"></span>Hoà {m["draw%"]}%</span>
                      <span className="prob-legend-item"><span className="legend-dot away"></span>Khách {m["away_win%"]}%</span>
                    </div>
                  </div>
                </div>

                {/* ── Form Section ── */}
                <div className="detail-section">
                  <div className="detail-section-title">
                    <span className="section-icon">📊</span>
                    Phong Độ Gần Đây
                  </div>
                  <div className="detail-form-row">
                    <div className="detail-form-side home">
                      {renderForm(m.home_form_str)}
                      <span className="form-pts">{homeFormPts} điểm / 5 trận</span>
                    </div>
                    <div className="detail-form-divider">5 TRẬN</div>
                    <div className="detail-form-side away">
                      {renderForm(m.away_form_str)}
                      <span className="form-pts">{awayFormPts} điểm / 5 trận</span>
                    </div>
                  </div>
                </div>

                {/* ── Stats Comparison ── */}
                <div className="detail-section">
                  <div className="detail-section-title">
                    <span className="section-icon">⚔️</span>
                    So Sánh Chỉ Số
                  </div>
                  <div className="detail-stats-grid">
                    <ComparisonBar homeVal={m.home_gf} awayVal={m.away_gf} label="Bàn thắng / trận" />
                    <ComparisonBar homeVal={m.home_ga} awayVal={m.away_ga} label="Bàn thua / trận" higherIsBetter={false} />
                    <ComparisonBar homeVal={m.home_cs} awayVal={m.away_cs} label="Tỉ lệ giữ sạch lưới" isPercentage={true} />
                    <ComparisonBar homeVal={m.home_elo} awayVal={m.away_elo} label="Điểm ELO" />
                  </div>
                </div>

                {/* ── H2H Section ── */}
                <div className="detail-section">
                  <div className="detail-section-title">
                    <span className="section-icon">🤝</span>
                    Lịch Sử Đối Đầu
                    <span className="h2h-sub">(6 trận gần nhất)</span>
                  </div>
                  <div className="h2h-visual">
                    <div className="h2h-bar-container">
                      <div className="h2h-bar-fill h2h-win" style={{ width: `${(h2hWins / h2hTotal) * 100}%` }}>
                        {h2hWins > 0 && <span>{h2hWins}</span>}
                      </div>
                      <div className="h2h-bar-fill h2h-draw" style={{ width: `${(h2hDraws / h2hTotal) * 100}%` }}>
                        {h2hDraws > 0 && <span>{h2hDraws}</span>}
                      </div>
                      <div className="h2h-bar-fill h2h-loss" style={{ width: `${(h2hLosses / h2hTotal) * 100}%` }}>
                        {h2hLosses > 0 && <span>{h2hLosses}</span>}
                      </div>
                    </div>
                    <div className="h2h-legend">
                      <span className="h2h-legend-item win">Thắng {h2hWins}</span>
                      <span className="h2h-legend-item draw">Hoà {h2hDraws}</span>
                      <span className="h2h-legend-item loss">Thua {h2hLosses}</span>
                    </div>
                    <div className="h2h-note">* Tính từ góc nhìn của {m.home}</div>
                  </div>
                </div>

              </div>
            </div>
          );
        })()}
      </div>
    );
  }