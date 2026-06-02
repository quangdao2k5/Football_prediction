import { useEffect, useState } from "react";

export default function Predictions({ data, apiBase, gameweeks = [] }) {
    const [selectedMatch, setSelectedMatch] = useState(null);
    const [currentData, setCurrentData] = useState(data);
    const [selectedGw, setSelectedGw] = useState(data?.gameweek || null);
    const [loadingGw, setLoadingGw] = useState(false);
    const [activeGroup, setActiveGroup] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
      setCurrentData(data);
      setSelectedGw(data?.gameweek || null);
      setActiveGroup(null);
    }, [data]);

    const availableGameweeks = gameweeks.length
      ? gameweeks
      : (data?.gameweek ? [data.gameweek] : []);

    const loadGameweek = async (gw) => {
      if (!apiBase || gw === selectedGw) return;

      setLoadingGw(true);
      setError(null);
      try {
        const res = await fetch(`${apiBase}/predictions/${gw}`);
        if (!res.ok) throw new Error(`Không tải được GW${gw}`);
        const next = await res.json();
        setCurrentData(next);
        setSelectedGw(gw);
        setSelectedMatch(null);
        setActiveGroup(null);
      } catch (err) {
        setError(err.message || "Không tải được dữ liệu gameweek");
      } finally {
        setLoadingGw(false);
      }
    };

    if (!currentData) return <div className="empty">Chưa có dữ liệu dự đoán.</div>;

    const { gameweek, matches = [] } = currentData;
  
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

    const shortResultLabel = (pred) => {
      if (pred === "Home Win") return "Nhà thắng";
      if (pred === "Away Win") return "Khách thắng";
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
      const probs = [
        { key: "H", label: "Nhà thắng", value: Number(m["home_win%"]) || 0 },
        { key: "D", label: "Hoà", value: Number(m["draw%"]) || 0 },
        { key: "A", label: "Khách thắng", value: Number(m["away_win%"]) || 0 },
      ].sort((a, b) => b.value - a.value);

      const maxProb = probs[0]?.value || 0;
      const runnerUp = probs[1] || { label: "lựa chọn khác", value: 0 };
      const margin = Number((maxProb - runnerUp.value).toFixed(1));
      const base = { maxProb, margin, leader: probs[0], runnerUp };

      if (maxProb >= 65 && margin >= 20) {
        return { ...base, level: "Rất cao", color: "var(--green)", className: "high" };
      }
      if (maxProb >= 55 && margin >= 12) {
        return { ...base, level: "Cao", color: "var(--accent)", className: "good" };
      }
      if (maxProb >= 45 && margin >= 8) {
        return { ...base, level: "Trung bình", color: "var(--yellow)", className: "medium" };
      }
      return { ...base, level: "Cân bằng", color: "var(--text-muted)", className: "low" };
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

    const formatNumber = (value, digits = 1) => {
      const num = Number(value) || 0;
      return num.toFixed(digits).replace(/\.0$/, "");
    };

    const formatPercent = (value) => `${formatNumber(value)}%`;

    const getMatchNotes = (m) => {
      const confidence = getConfidence(m);
      const homeFormPts = getFormPoints(m.home_form_str);
      const awayFormPts = getFormPoints(m.away_form_str);
      const eloDiff = Math.round((Number(m.home_elo) || 0) - (Number(m.away_elo) || 0));
      const gfDiff = Number(((Number(m.home_gf) || 0) - (Number(m.away_gf) || 0)).toFixed(1));
      const drawProb = Number(m["draw%"]) || 0;
      const eloNote = eloDiff > 0
        ? `${m.home} cao hơn ${m.away} ${Math.abs(eloDiff)} điểm ELO.`
        : eloDiff < 0
          ? `${m.away} cao hơn ${m.home} ${Math.abs(eloDiff)} điểm ELO.`
          : `${m.home} và ${m.away} đang ngang bằng ELO.`;
      const scoringNote = gfDiff > 0
        ? `${m.home} ghi nhiều hơn ${m.away} ${formatNumber(Math.abs(gfDiff))} bàn/trận gần đây.`
        : gfDiff < 0
          ? `${m.away} ghi nhiều hơn ${m.home} ${formatNumber(Math.abs(gfDiff))} bàn/trận gần đây.`
          : `${m.home} và ${m.away} có hiệu suất ghi bàn gần đây ngang nhau.`;

      const notes = [
        `Dự đoán chính: ${confidence.leader.label} với xác suất ${formatPercent(confidence.maxProb)}. Xác suất chi tiết: Nhà ${formatPercent(m["home_win%"])}, Hoà ${formatPercent(m["draw%"])}, Khách ${formatPercent(m["away_win%"])}.`,
        `Phong độ 5 trận gần nhất: ${m.home} giành ${homeFormPts} điểm, ${m.away} giành ${awayFormPts} điểm.`,
        eloNote,
        scoringNote,
      ];

      if (drawProb >= 25) {
        notes.push(`Cửa hoà đáng chú ý ở mức ${formatPercent(drawProb)}, nên trận này không quá một chiều.`);
      }

      return notes.slice(0, 4);
    };

    const recordLabel = (record) => {
      if (!record || !record.played) return "Chưa đủ dữ liệu";
      return `${record.wins}T ${record.draws}H ${record.losses}B`;
    };

    const pctOrDash = (value) => {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
      return `${Math.round(Number(value) * 100)}%`;
    };

    const ResultPill = ({ result }) => (
      <span className={`result-pill ${result}`}>{result}</span>
    );

    const RecentMatchList = ({ matches: recentMatches = [] }) => {
      if (!recentMatches.length) {
        return <div className="mini-empty">Chưa đủ dữ liệu trước trận.</div>;
      }

      return (
        <div className="recent-list">
          {recentMatches.slice().reverse().map((item, idx) => (
            <div className="recent-item" key={`${item.date}-${item.opponent}-${idx}`}>
              <ResultPill result={item.result} />
              <span className="recent-opponent">
                <span className="recent-venue">{item.venue === "H" ? "Sân nhà" : "Sân khách"}</span>
                <span>{item.opponent}</span>
              </span>
              <span className="recent-score">{item.score}</span>
            </div>
          ))}
        </div>
      );
    };

    const H2HMatchList = ({ matches: h2hMatches = [], team }) => {
      if (!h2hMatches.length) {
        return <div className="mini-empty">Chưa có trận đối đầu gần đây trong dữ liệu trước trận.</div>;
      }

      const resultText = {
        W: "thắng",
        D: "hoà",
        L: "thua",
      };

      const tagLabel = h2hMatches.length === 1
        ? "Lần đối đầu gần nhất"
        : "Lần gặp trước";

      return (
        <div className="h2h-match-list">
          {h2hMatches.slice().reverse().map((item, idx) => (
            <div className="h2h-match-row" key={`${item.date}-${item.opponent}-${idx}`}>
              <span className={`history-tag ${item.result}`}>{tagLabel}</span>
              <span className="h2h-match-text">
                <strong>{team} {resultText[item.result] || ""} {item.opponent} {item.score}</strong>
                <small>{item.date} · {team} đá {item.venue === "H" ? "sân nhà" : "sân khách"} ở trận đó</small>
              </span>
            </div>
          ))}
        </div>
      );
    };

    const VenueRecordCard = ({ title, team, record }) => (
      <div className="venue-card">
        <div className="venue-card-head">
          <div>
            <div className="venue-title">{title}</div>
            <div className="venue-team">{team}</div>
          </div>
          <div className="venue-record">{recordLabel(record)}</div>
        </div>
        <div className="venue-metrics">
          <span><strong>{record?.ppg ?? 0}</strong> điểm/trận</span>
          <span><strong>{record?.gf_avg ?? 0}</strong> bàn ghi/trận</span>
          <span><strong>{record?.ga_avg ?? 0}</strong> bàn thua/trận</span>
          <span><strong>{record?.clean_sheets ?? 0}</strong> sạch lưới</span>
        </div>
        {renderForm(record?.form)}
        <RecentMatchList matches={record?.matches || []} />
      </div>
    );

    const isClearTrend = (m) => {
      const confidence = getConfidence(m);
      return confidence.className === "high" || confidence.className === "good";
    };

    const isBalancedMatch = (m) => getConfidence(m).className === "low";

    const isDrawWatch = (m) => (Number(m["draw%"]) || 0) >= 25;

    const getGroupReason = (m, groupId) => {
      const confidence = getConfidence(m);
      if (groupId === "clear") {
        return `Độ tin cậy ${confidence.level.toLowerCase()}, xác suất chọn ${formatPercent(confidence.maxProb)}.`;
      }
      if (groupId === "balanced") {
        return `Ba cửa khá sát nhau: Nhà ${formatPercent(m["home_win%"])}, Hoà ${formatPercent(m["draw%"])}, Khách ${formatPercent(m["away_win%"])}.`;
      }
      if (groupId === "draw") {
        return `Xác suất hoà ${formatPercent(m["draw%"])}.`;
      }
      return "";
    };

    const summary = matches.reduce((acc, m) => {
      const confidence = getConfidence(m);
      acc.maxProbSum += confidence.maxProb;
      if (isClearTrend(m)) acc.clear += 1;
      if (isBalancedMatch(m)) acc.balanced += 1;
      if (isDrawWatch(m)) acc.drawWatch += 1;
      if (m.prediction === "Home Win") acc.home += 1;
      else if (m.prediction === "Away Win") acc.away += 1;
      else acc.draw += 1;
      return acc;
    }, { maxProbSum: 0, clear: 0, balanced: 0, drawWatch: 0, home: 0, away: 0, draw: 0 });

    const avgConfidence = matches.length
      ? Math.round(summary.maxProbSum / matches.length)
      : 0;

    const groupConfig = {
      clear: {
        title: "Trận rõ xu hướng",
        count: summary.clear,
        filter: isClearTrend,
      },
      balanced: {
        title: "Trận cân bằng",
        count: summary.balanced,
        filter: isBalancedMatch,
      },
      draw: {
        title: "Cửa hoà đáng chú ý",
        count: summary.drawWatch,
        filter: isDrawWatch,
      },
    };

    const activeGroupConfig = activeGroup ? groupConfig[activeGroup] : null;
    const activeGroupMatches = activeGroupConfig
      ? matches.filter(activeGroupConfig.filter)
      : [];

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
	          <div>
	            <h2>Dự đoán Gameweek {gameweek}</h2>
	            <p className="section-subtitle">Các xác suất được tính từ dữ liệu trước trận: phong độ, ELO, hiệu suất ghi bàn, phòng ngự và đối đầu.</p>
	          </div>
	          <span className="badge">{matches.length} trận</span>
	        </div>

	        {availableGameweeks.length > 1 && (
	          <div className="gw-selector" aria-label="Chọn gameweek">
	            {availableGameweeks.map(gw => (
	              <button
	                key={gw}
	                className={`gw-btn ${gw === gameweek ? "active" : ""}`}
	                onClick={() => loadGameweek(gw)}
	                disabled={loadingGw}
	              >
	                GW{gw}
	              </button>
	            ))}
	          </div>
	        )}

	        {error && <div className="inline-error">{error}</div>}

	        <div className="prediction-overview">
	          <div className="metric-tile">
	            <span className="metric-value">{avgConfidence}%</span>
	            <span
	              className="metric-label"
	              title="Trung bình xác suất của phương án được model chọn ở mỗi trận."
	            >
	              Độ tự tin dự đoán TB
	            </span>
	          </div>
	          <button
	            type="button"
	            className={`metric-tile metric-action ${activeGroup === "clear" ? "active" : ""}`}
	            onClick={() => setActiveGroup(activeGroup === "clear" ? null : "clear")}
	          >
	            <span className="metric-value">{summary.clear}</span>
	            <span className="metric-label">Trận rõ xu hướng</span>
	          </button>
	          <button
	            type="button"
	            className={`metric-tile metric-action ${activeGroup === "balanced" ? "active" : ""}`}
	            onClick={() => setActiveGroup(activeGroup === "balanced" ? null : "balanced")}
	          >
	            <span className="metric-value">{summary.balanced}</span>
	            <span className="metric-label">Trận cân bằng</span>
	          </button>
	          <button
	            type="button"
	            className={`metric-tile metric-action ${activeGroup === "draw" ? "active" : ""}`}
	            onClick={() => setActiveGroup(activeGroup === "draw" ? null : "draw")}
	          >
	            <span className="metric-value">{summary.drawWatch}</span>
	            <span className="metric-label">Cửa hoà đáng chú ý</span>
	          </button>
	          <div className="metric-tile wide">
	            <span className="metric-value compact">{summary.home}H · {summary.draw}D · {summary.away}A</span>
	            <span className="metric-label">Phân bố dự đoán</span>
	          </div>
	        </div>

	        {activeGroupConfig && (
	          <div className="group-panel">
	            <div className="group-panel-header">
	              <div>
	                <h3>{activeGroupConfig.title}</h3>
	                <p>{activeGroupMatches.length} trận thuộc nhóm này trong GW{gameweek}</p>
	              </div>
	              <button type="button" className="group-close" onClick={() => setActiveGroup(null)}>
	                Đóng
	              </button>
	            </div>
	            {activeGroupMatches.length > 0 ? (
	              <div className="group-match-list">
	                {activeGroupMatches.map((m) => {
	                  const confidence = getConfidence(m);
	                  return (
	                    <button
	                      type="button"
	                      className="group-match-row"
	                      key={`${activeGroup}-${m.home}-${m.away}`}
	                      onClick={() => setSelectedMatch(m)}
	                    >
	                      <span className={`confidence-chip ${confidence.className}`}>{confidence.level}</span>
	                      <span className="group-match-teams">{m.home} vs {m.away}</span>
	                      <span className="group-match-pred">{resultLabel(m.prediction)}</span>
	                      <span className="group-match-reason">{getGroupReason(m, activeGroup)}</span>
	                    </button>
	                  );
	                })}
	              </div>
	            ) : (
	              <div className="mini-empty">Không có trận nào trong nhóm này.</div>
	            )}
	          </div>
	        )}
	  
	        <div className="match-grid">
	          {matches.map((m, i) => {
	            const confidence = getConfidence(m);
	            const isActiveGroupMatch = activeGroupConfig?.filter(m);
	            return (
	              <div
	                className={`match-card clickable ${confidence.margin <= 10 ? "balanced" : ""} ${isActiveGroupMatch ? "group-highlight" : ""} ${activeGroup && !isActiveGroupMatch ? "group-dim" : ""}`}
	                key={`${gameweek}-${m.home}-${m.away}`}
	                onClick={() => setSelectedMatch(m)}
	              >
	                <div className="match-topline">
	                  <div className="match-date">{m.date}</div>
	                  <span className={`confidence-chip ${confidence.className}`}>{confidence.level}</span>
	                </div>
	  
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
	                  {resultLabel(m.prediction)}
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
          const matchNotes = getMatchNotes(m);

          const h2hMatches = m.h2h_matches || [];

          // Parse H2H string as fallback when detailed H2H rows are unavailable.
          const h2hMatch = (m.h2h_str || "").match(/Thắng (\d+).*Hòa (\d+).*Thua (\d+)/);
          const h2hWins = h2hMatches.length
            ? h2hMatches.filter(item => item.result === "W").length
            : h2hMatch ? parseInt(h2hMatch[1]) : 0;
          const h2hDraws = h2hMatches.length
            ? h2hMatches.filter(item => item.result === "D").length
            : h2hMatch ? parseInt(h2hMatch[2]) : 0;
          const h2hLosses = h2hMatches.length
            ? h2hMatches.filter(item => item.result === "L").length
            : h2hMatch ? parseInt(h2hMatch[3]) : 0;
          const h2hTotal = h2hWins + h2hDraws + h2hLosses || 1;
          const h2hLabel = h2hTotal >= 6
            ? "6 trận gần nhất"
            : `${h2hTotal} trận gần nhất có dữ liệu`;

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

	                <div className="detail-section">
	                  <div className="detail-section-title">
	                    Yếu tố trước trận
	                  </div>
	                  <div className="factor-list">
	                    {matchNotes.map((note, idx) => (
	                      <div className="factor-item" key={idx}>{note}</div>
	                    ))}
	                  </div>
	                </div>

                {(m.home_position || m.away_position || m.home_points || m.away_points) && (
                  <div className="detail-section">
                    <div className="detail-section-title">
                      Vị trí trước trận
                    </div>
                    <div className="table-context-grid">
                      <div className="table-context-card">
                        <span className="context-team">{m.home}</span>
                        <strong>#{m.home_position ?? "—"}</strong>
                        <span>{m.home_points ?? "—"} điểm</span>
                        <small>Động lực {pctOrDash(m.home_motivation)}</small>
                      </div>
                      <div className="table-context-card">
                        <span className="context-team">{m.away}</span>
                        <strong>#{m.away_position ?? "—"}</strong>
                        <span>{m.away_points ?? "—"} điểm</span>
                        <small>Động lực {pctOrDash(m.away_motivation)}</small>
                      </div>
                    </div>
                  </div>
                )}

	                {/* ── Form Section ── */}
	                <div className="detail-section">
	                  <div className="detail-section-title">
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

                <div className="detail-section">
                  <div className="detail-section-title">
                    Sân nhà / sân khách
                  </div>
                  <div className="venue-grid">
                    <VenueRecordCard
                      title="Sân nhà gần đây"
                      team={m.home}
                      record={m.home_home_record}
                    />
                    <VenueRecordCard
                      title="Sân khách gần đây"
                      team={m.away}
                      record={m.away_away_record}
                    />
                  </div>
                </div>

                {/* ── Stats Comparison ── */}
	                <div className="detail-section">
	                  <div className="detail-section-title">
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
	                    Lịch Sử Đối Đầu
                    <span className="h2h-sub">({h2hLabel})</span>
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
                    <div className="h2h-note">* Thống kê tính từ góc nhìn của {m.home}.</div>
                    <H2HMatchList matches={h2hMatches} team={m.home} />
                  </div>
                </div>

              </div>
            </div>
          );
        })()}
      </div>
    );
  }
