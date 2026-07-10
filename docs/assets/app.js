const DATA_URL = "data/task2_indicators.csv";

const colors = {
  text: "#f3f4f8",
  muted: "#a8adba",
  grid: "rgba(255,255,255,0.1)",
  blue: "#5ca7ff",
  gold: "#f4bd3d",
  green: "#64d08b",
  red: "#ff6b6b",
  purple: "#a788ff",
  white: "#f5f5f5",
};

function parseCsv(text) {
  const [headerLine, ...lines] = text.trim().split(/\r?\n/);
  const headers = headerLine.split(",");
  return lines.map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => {
      const raw = values[index] ?? "";
      const value = raw === "" ? null : Number(raw);
      return [header, Number.isNaN(value) ? raw : value];
    }));
  });
}

function dateLabel(value) {
  const text = String(value);
  return `${text.slice(0, 4)}-${text.slice(4, 6)}-${text.slice(6, 8)}`;
}

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function latestSignal(row) {
  const rsiSignal = row.rsi14 >= 70 ? "偏强" : row.rsi14 <= 30 ? "偏弱" : "中性";
  const macdSignal = row.macd_hist > 0 ? "买入" : "卖出";
  const bandSignal = row.close > row.bb_upper ? "偏高" : row.close < row.bb_lower ? "偏低" : "观察";
  const score = [
    row.rsi14 >= 55 ? 1 : row.rsi14 <= 45 ? -1 : 0,
    row.macd_hist > 0 ? 1 : -1,
    row.close >= row.bb_middle ? 1 : -1,
    row.kdj_k > row.kdj_d ? 1 : -1,
  ].reduce((sum, value) => sum + value, 0);

  setText("rsi-signal", rsiSignal);
  setText("rsi-detail", `RSI ${row.rsi14.toFixed(2)}`);
  setText("macd-signal", macdSignal);
  setText("macd-detail", `MACD柱 ${row.macd_hist.toFixed(3)}`);
  setText("band-signal", bandSignal);
  setText("band-detail", `收盘 ${row.close.toFixed(2)} / 中轨 ${row.bb_middle.toFixed(2)}`);
  setText("score-value", score > 1 ? "偏强" : score < -1 ? "偏弱" : "中性");
  setText("score-detail", `综合分 ${score > 0 ? "+" : ""}${score}`);
  setText("sample-count", String(row.__count));
  setText("latest-close", row.close.toFixed(2));
  setText("latest-date", dateLabel(row.trade_date).slice(5));
}

function setupCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(320, Math.floor(rect.width * ratio));
  canvas.height = Math.max(220, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: canvas.width / ratio, height: canvas.height / ratio };
}

function seriesBounds(rows, keys, fixed) {
  if (fixed) return fixed;
  const values = rows.flatMap((row) => keys.map((key) => row[key]).filter((value) => typeof value === "number" && Number.isFinite(value)));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = (max - min || 1) * 0.08;
  return [min - pad, max + pad];
}

function chartScales(rows, keys, canvas, fixed) {
  const { ctx, width, height } = setupCanvas(canvas);
  const margin = { left: 48, right: 18, top: 18, bottom: 34 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const [minY, maxY] = seriesBounds(rows, keys, fixed);
  const x = (index) => margin.left + (innerW * index) / Math.max(rows.length - 1, 1);
  const y = (value) => margin.top + innerH * (1 - (value - minY) / (maxY - minY || 1));
  return { ctx, width, height, margin, innerW, innerH, minY, maxY, x, y };
}

function drawGrid(state, rows) {
  const { ctx, width, height, margin, minY, maxY, y } = state;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "rgba(255,255,255,0.02)";
  ctx.fillRect(margin.left, margin.top, state.innerW, state.innerH);
  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  ctx.font = "12px -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.fillStyle = colors.muted;

  for (let i = 0; i <= 4; i += 1) {
    const value = minY + ((maxY - minY) * i) / 4;
    const yy = y(value);
    ctx.beginPath();
    ctx.moveTo(margin.left, yy);
    ctx.lineTo(width - margin.right, yy);
    ctx.stroke();
    ctx.fillText(value.toFixed(2), 8, yy + 4);
  }

  [0, Math.floor(rows.length / 2), rows.length - 1].forEach((index) => {
    const label = dateLabel(rows[index].trade_date).slice(2);
    ctx.fillText(label, state.x(index) - 24, height - 10);
  });
}

function drawLineSpecs(state, rows, specs) {
  specs.forEach((spec) => {
    state.ctx.strokeStyle = spec.color;
    state.ctx.lineWidth = spec.width || 2;
    state.ctx.beginPath();
    let started = false;
    rows.forEach((row, index) => {
      const value = row[spec.key];
      if (value == null || !Number.isFinite(value)) {
        started = false;
        return;
      }
      const pointX = state.x(index);
      const pointY = state.y(value);
      if (!started) {
        state.ctx.moveTo(pointX, pointY);
        started = true;
      } else {
        state.ctx.lineTo(pointX, pointY);
      }
    });
    state.ctx.stroke();
  });
}

function drawLines(canvas, rows, specs, fixed) {
  const keys = specs.map((spec) => spec.key);
  const state = chartScales(rows, keys, canvas, fixed);
  drawGrid(state, rows);
  drawLineSpecs(state, rows, specs);
}

function drawRsi(canvas, rows) {
  const state = chartScales(rows, ["rsi14"], canvas, [0, 100]);
  drawGrid(state, rows);
  [30, 70].forEach((level) => {
    state.ctx.strokeStyle = level === 70 ? colors.red : colors.green;
    state.ctx.setLineDash([5, 5]);
    state.ctx.beginPath();
    state.ctx.moveTo(state.margin.left, state.y(level));
    state.ctx.lineTo(state.width - state.margin.right, state.y(level));
    state.ctx.stroke();
    state.ctx.setLineDash([]);
  });
  drawLineSpecs(state, rows, [{ key: "rsi14", color: colors.blue, width: 2 }]);
}

function drawMacd(canvas, rows) {
  const state = chartScales(rows, ["macd_dif", "macd_dea", "macd_hist"], canvas);
  drawGrid(state, rows);
  const zeroY = state.y(0);
  const barW = Math.max(1, state.innerW / rows.length * 0.7);
  rows.forEach((row, index) => {
    const value = row.macd_hist;
    if (value == null || !Number.isFinite(value)) return;
    const xx = state.x(index) - barW / 2;
    const yy = state.y(value);
    state.ctx.fillStyle = value >= 0 ? "rgba(255, 107, 107, 0.65)" : "rgba(100, 208, 139, 0.65)";
    state.ctx.fillRect(xx, Math.min(yy, zeroY), barW, Math.abs(zeroY - yy));
  });
  drawLineSpecs(state, rows, [
    { key: "macd_dif", color: colors.blue, width: 2 },
    { key: "macd_dea", color: colors.gold, width: 2 },
  ]);
}

function render(rows) {
  const latest = { ...rows[rows.length - 1], __count: rows.length };
  latestSignal(latest);
  drawLines(document.getElementById("price-chart"), rows, [{ key: "close", color: colors.blue, width: 2 }]);
  drawRsi(document.getElementById("rsi-chart"), rows);
  drawMacd(document.getElementById("macd-chart"), rows);
  drawLines(document.getElementById("bollinger-chart"), rows, [
    { key: "close", color: colors.white, width: 2 },
    { key: "bb_middle", color: colors.blue, width: 2 },
    { key: "bb_upper", color: colors.red, width: 1.5 },
    { key: "bb_lower", color: colors.green, width: 1.5 },
  ]);
  drawLines(document.getElementById("kdj-chart"), rows, [
    { key: "kdj_k", color: colors.blue, width: 2 },
    { key: "kdj_d", color: colors.gold, width: 2 },
    { key: "kdj_j", color: colors.purple, width: 2 },
  ], [0, 110]);
}

async function init() {
  const response = await fetch(DATA_URL);
  const rows = parseCsv(await response.text());
  render(rows);
  window.addEventListener("resize", () => render(rows));
}

init().catch((error) => {
  console.error(error);
});
