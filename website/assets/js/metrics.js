async function fetchText(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`);
  }
  return response.text();
}

function parseCSVLine(line) {
  const out = [];
  let value = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];

    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        value += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (ch === "," && !inQuotes) {
      out.push(value);
      value = "";
      continue;
    }

    value += ch;
  }

  out.push(value);
  return out;
}

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (!lines.length) return [];

  const headers = parseCSVLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = parseCSVLine(line);
    const row = {};
    headers.forEach((h, i) => {
      row[h] = values[i] ?? "";
    });
    return row;
  });
}

function asNum(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}

function fmt(n, digits = 4) {
  if (!Number.isFinite(n)) return "N/A";
  return n.toFixed(digits);
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function renderModelTable(rows) {
  const body = document.getElementById("model-table-body");
  if (!body) return;
  body.innerHTML = "";

  rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.model}</td>
      <td>${fmt(asNum(r.RMSE), 4)}</td>
      <td>${fmt(asNum(r.MAE), 4)}</td>
      <td>${fmt(asNum(r.R2), 4)}</td>
    `;
    body.appendChild(tr);
  });
}

function renderArimaTable(rows) {
  const body = document.getElementById("arima-table-body");
  if (!body) return;
  body.innerHTML = "";

  rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.model}</td>
      <td>${fmt(asNum(r.RMSE), 4)}</td>
    `;
    body.appendChild(tr);
  });
}

function renderTopFeatures(rows, limit = 8) {
  const list = document.getElementById("top-features");
  if (!list) return;
  list.innerHTML = "";

  rows
    .slice()
    .sort((a, b) => asNum(b.mutual_info) - asNum(a.mutual_info))
    .slice(0, limit)
    .forEach((r, idx) => {
      const li = document.createElement("li");
      li.textContent = `${idx + 1}. ${r.feature} (MI=${fmt(asNum(r.mutual_info), 3)})`;
      list.appendChild(li);
    });
}

async function loadPanelData() {
  const status = document.getElementById("data-status");

  try {
    const [modelText, arimaText, featureText] = await Promise.all([
      fetchText("../outputs/tables/model_metrics.csv"),
      fetchText("../outputs/tables/arima_metrics.csv"),
      fetchText("../outputs/tables/feature_selection_mi.csv")
    ]);

    const modelRows = parseCSV(modelText);
    const arimaRows = parseCSV(arimaText);
    const featureRows = parseCSV(featureText);

    if (status) {
      status.textContent = "Live metrics loaded from outputs/tables.";
    }

    const mlRows = modelRows.filter((r) => r.model !== "SeasonalityOnly");
    const best = mlRows
      .slice()
      .sort((a, b) => asNum(a.RMSE) - asNum(b.RMSE))[0];
    const seasonal = modelRows.find((r) => r.model === "SeasonalityOnly");
    const arima = arimaRows.find((r) => r.model.includes("ARIMA"));
    const naive = arimaRows.find((r) => r.model.includes("SeasonalNaive"));

    if (best) {
      setText("best-model-name", best.model);
      setText("best-model-rmse", fmt(asNum(best.RMSE)));
      setText("best-model-mae", fmt(asNum(best.MAE)));
      setText("best-model-r2", fmt(asNum(best.R2)));
    }

    if (seasonal) {
      setText("seasonal-rmse", fmt(asNum(seasonal.RMSE)));
      setText("seasonal-r2", fmt(asNum(seasonal.R2)));
    }

    if (arima) {
      setText("arima-rmse", fmt(asNum(arima.RMSE)));
    }

    if (arima && naive) {
      const a = asNum(arima.RMSE);
      const n = asNum(naive.RMSE);
      const improvement = Number.isFinite(a) && Number.isFinite(n) && n !== 0
        ? ((n - a) / n) * 100
        : NaN;
      setText("arima-improvement", Number.isFinite(improvement) ? `${improvement.toFixed(1)}%` : "N/A");
    }

    renderModelTable(modelRows);
    renderArimaTable(arimaRows);
    renderTopFeatures(featureRows);
  } catch (err) {
    if (status) {
      status.textContent = "Could not load output tables. Serve this folder from the project root with `python -m http.server` and open /website/index.html.";
    }
    console.error(err);
  }
}

window.addEventListener("DOMContentLoaded", loadPanelData);
