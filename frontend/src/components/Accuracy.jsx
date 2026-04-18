import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer } from "recharts";

export default function Accuracy({ data }) {
  if (!data || data.history.length === 0)
    return <div className="empty">Chưa có dữ liệu accuracy. Chạy thêm vài gameweek nhé!</div>;

  const { history, overall, total_correct, total_matches } = data;

  const chartData = history.map(row => ({
    gw:       `GW${row.gameweek}`,
    accuracy: Math.round(row.accuracy * 100),
    correct:  row.correct,
    total:    row.total,
  }));

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0].payload;
    return (
      <div className="tooltip">
        <p className="tooltip-title">{label}</p>
        <p>Accuracy: <strong>{d.accuracy}%</strong></p>
        <p>Đúng: <strong>{d.correct}/{d.total}</strong></p>
      </div>
    );
  };

  return (
    <div className="section">
      <div className="section-header">
        <h2>Lịch sử độ chính xác</h2>
      </div>

      <div className="stats-row">
        <div className="stat-card">
          <div className="stat-value">{Math.round((overall || 0) * 100)}%</div>
          <div className="stat-label">Accuracy tích lũy</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{total_correct}</div>
          <div className="stat-label">Trận dự đoán đúng</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{total_matches}</div>
          <div className="stat-label">Tổng trận đã đánh giá</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{history.length}</div>
          <div className="stat-label">Gameweeks đã log</div>
        </div>
      </div>

      <div className="chart-container">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="gw" tick={{ fill: "var(--text-muted)", fontSize: 12 }} />
            <YAxis
              domain={[0, 100]}
              tickFormatter={v => `${v}%`}
              tick={{ fill: "var(--text-muted)", fontSize: 12 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine
              y={Math.round((overall || 0) * 100)}
              stroke="var(--accent)"
              strokeDasharray="4 4"
              label={{ value: "Trung bình", fill: "var(--accent)", fontSize: 11 }}
            />
            <Line
              type="monotone"
              dataKey="accuracy"
              stroke="var(--accent)"
              strokeWidth={2}
              dot={{ fill: "var(--accent)", r: 4 }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="accuracy-table">
        <table>
          <thead>
            <tr>
              <th>Gameweek</th>
              <th>Đúng</th>
              <th>Tổng</th>
              <th>Accuracy</th>
            </tr>
          </thead>
          <tbody>
            {[...history].reverse().map(row => (
              <tr key={row.gameweek}>
                <td>GW{row.gameweek}</td>
                <td>{row.correct}</td>
                <td>{row.total}</td>
                <td>
                  <span className={`acc-badge ${row.accuracy >= 0.5 ? "good" : "bad"}`}>
                    {Math.round(row.accuracy * 100)}%
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}