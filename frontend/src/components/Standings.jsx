export default function Standings({ data }) {
    if (!data || !data.standings?.length)
      return <div className="empty">Chưa có dữ liệu bảng xếp hạng.</div>;
  
    const { standings, source } = data;
  
    const posColor = (pos) => {
      if (pos <= 4)  return "var(--green)";   // Top 4 - Champions League
      if (pos <= 6)  return "var(--blue)";    // Europa League
      if (pos >= 18) return "var(--red)";     // Relegation
      return "transparent";
    };
  
    return (
      <div className="section">
        <div className="section-header">
          <h2>Bảng xếp hạng EPL 2025/26</h2>
          <span className="badge">{source === "api" ? "Live" : "Tính từ dữ liệu local"}</span>
        </div>
  
        <div className="legend">
          <span><span className="dot" style={{background:"var(--green)"}}/>Champions League</span>
          <span><span className="dot" style={{background:"var(--blue)"}}/>Europa League</span>
          <span><span className="dot" style={{background:"var(--red)"}}/>Xuống hạng</span>
        </div>
  
        <div className="table-wrap">
          <table className="standings-table">
            <thead>
              <tr>
                <th>#</th>
                <th className="left">Đội</th>
                <th>Trận</th>
                <th>T</th>
                <th>H</th>
                <th>B</th>
                <th>BT</th>
                <th>BB</th>
                <th>HS</th>
                <th>Điểm</th>
              </tr>
            </thead>
            <tbody>
              {standings.map((row, i) => (
                <tr key={i} className={i % 2 === 0 ? "row-even" : ""}>
                  <td>
                    <span className="pos-indicator" style={{ borderColor: posColor(row.position) }}>
                      {row.position}
                    </span>
                  </td>
                  <td className="team-cell left">{row.team}</td>
                  <td>{row.played}</td>
                  <td>{row.won}</td>
                  <td>{row.drawn}</td>
                  <td>{row.lost}</td>
                  <td>{row.goals_for}</td>
                  <td>{row.goals_against}</td>
                  <td className={row.goal_diff > 0 ? "positive" : row.goal_diff < 0 ? "negative" : ""}>
                    {row.goal_diff > 0 ? `+${row.goal_diff}` : row.goal_diff}
                  </td>
                  <td><strong>{row.points}</strong></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }