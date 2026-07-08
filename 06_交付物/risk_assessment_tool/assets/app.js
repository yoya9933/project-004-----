const DATA_URL = "data/risk_tool_data.json";

const state = {
  data: null,
  filteredCases: [],
  selectedJid: null,
  contributionModel: "logistic_regression_l2",
};

const riskClass = {
  高: "risk-high",
  中: "risk-mid",
  低: "risk-low",
};

const riskRank = {
  高: 3,
  中: 2,
  低: 1,
};

const modelNames = {
  mean: "Mean baseline",
  ridge: "Ridge",
  lasso: "Lasso",
  logistic_regression_l2: "分類",
  ridge_regression_l2: "Ridge",
  lasso_regression_l1: "Lasso",
};

function $(id) {
  return document.getElementById(id);
}

function pct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function num(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function yesNo(value) {
  if (value === 1) return "是";
  if (value === 0) return "否";
  return "—";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function splitShort(split) {
  return {
    train_2021_2023: "訓練",
    validation_2024: "驗證",
    test_2025: "測試",
    latest_2026: "最新",
  }[split] || split || "—";
}

function splitLong(split) {
  return {
    train_2021_2023: "訓練集 2021-2023",
    validation_2024: "驗證集 2024",
    test_2025: "測試集 2025",
    latest_2026: "最新年度 2026",
  }[split] || split || "—";
}

function populateSelect(select, values, allLabel, formatter = (value) => value) {
  select.innerHTML = "";
  const all = document.createElement("option");
  all.value = "";
  all.textContent = allLabel;
  select.appendChild(all);
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = formatter(value);
    select.appendChild(option);
  });
}

function initFilters() {
  const cases = state.data.cases;
  const years = [...new Set(cases.map((item) => String(item.year)).filter(Boolean))].sort();
  const splits = [...new Set(cases.map((item) => item.split).filter(Boolean))];
  const risks = ["高", "中", "低"];

  populateSelect($("yearFilter"), years, "全部年度");
  populateSelect($("splitFilter"), splits, "全部切分", splitLong);
  populateSelect($("riskFilter"), risks, "全部風險");

  ["searchInput", "yearFilter", "splitFilter", "riskFilter"].forEach((id) => {
    $(id).addEventListener("input", applyFilters);
    $(id).addEventListener("change", applyFilters);
  });

  document.querySelectorAll(".segment").forEach((button) => {
    button.addEventListener("click", () => {
      state.contributionModel = button.dataset.model;
      document.querySelectorAll(".segment").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      renderSelectedCase();
    });
  });

  document.querySelectorAll(".demo-actions button").forEach((button) => {
    button.addEventListener("click", () => applyPreset(button.dataset.preset));
  });
}

function applyPreset(preset) {
  $("searchInput").value = "";
  $("yearFilter").value = "";
  $("splitFilter").value = "";
  $("riskFilter").value = "";

  if (preset === "high") {
    $("riskFilter").value = "高";
  }
  if (preset === "test2025") {
    $("yearFilter").value = "2025";
    $("splitFilter").value = "test_2025";
  }
  if (preset === "latest2026") {
    $("yearFilter").value = "2026";
    $("splitFilter").value = "latest_2026";
  }
  applyFilters();
}

function applyFilters() {
  const keyword = $("searchInput").value.trim().toLowerCase();
  const year = $("yearFilter").value;
  const split = $("splitFilter").value;
  const risk = $("riskFilter").value;

  state.filteredCases = state.data.cases
    .filter((item) => {
      const haystack = `${item.jid} ${item.title} ${item.court}`.toLowerCase();
      return (
        (!keyword || haystack.includes(keyword)) &&
        (!year || String(item.year) === year) &&
        (!split || item.split === split) &&
        (!risk || item.riskLevel === risk)
      );
    })
    .sort((a, b) => {
      const riskDiff = (riskRank[b.riskLevel] || 0) - (riskRank[a.riskLevel] || 0);
      if (riskDiff !== 0) return riskDiff;
      return (b.year || 0) - (a.year || 0);
    });

  if (!state.filteredCases.some((item) => item.jid === state.selectedJid)) {
    state.selectedJid = state.filteredCases[0]?.jid || null;
  }
  renderCaseList();
  renderSelectedCase();
}

function renderCaseList() {
  $("caseCount").textContent = `${state.filteredCases.length} 件`;
  $("selectedHint").textContent = state.selectedJid ? "已選取" : "未選取";

  if (!state.filteredCases.length) {
    $("caseList").innerHTML = `<div class="empty-state">沒有符合條件的案件</div>`;
    return;
  }

  $("caseList").innerHTML = state.filteredCases
    .map((item) => {
      const active = item.jid === state.selectedJid ? " active" : "";
      return `
        <button class="case-button${active}" type="button" data-jid="${escapeHtml(item.jid)}">
          <span class="case-button-title">${escapeHtml(item.title || "未命名案件")}</span>
          <span class="case-button-meta">
            <span>${escapeHtml(item.year)}｜${escapeHtml(splitShort(item.split))}</span>
            <span>${escapeHtml(item.riskLevel)}風險</span>
          </span>
          <span class="case-button-score">
            <span>${escapeHtml(item.court)}</span>
            <span>${pct(item.reductionProbability, 0)}</span>
          </span>
        </button>
      `;
    })
    .join("");

  document.querySelectorAll(".case-button").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedJid = button.dataset.jid;
      renderCaseList();
      renderSelectedCase();
    });
  });
}

function getSelectedCase() {
  return state.data.cases.find((item) => item.jid === state.selectedJid) || null;
}

function setRiskBadge(item) {
  const badge = $("riskBadge");
  badge.className = `risk-badge ${riskClass[item.riskLevel] || "risk-low"}`;
  badge.textContent = `${item.riskLevel}風險`;
}

function renderSelectedCase() {
  const item = getSelectedCase();
  if (!item) {
    $("caseTitle").textContent = "沒有符合條件的案件";
    $("caseMeta").textContent = "-";
    $("caseJid").textContent = "-";
    $("riskBadge").textContent = "-";
    return;
  }

  $("caseMeta").textContent = `${item.year}｜${item.splitLabel}｜${item.court}`;
  $("caseTitle").textContent = item.title || "未命名案件";
  $("caseJid").textContent = item.jid;
  setRiskBadge(item);

  $("probabilityMetric").textContent = pct(item.reductionProbability);
  $("remainingMetric").textContent = pct(item.ridgePredictedRemainingRatio);
  $("reductionMetric").textContent = pct(item.ridgePredictedReductionRate);
  $("bucketMetric").textContent = item.ridgePredictedBucket || "無比例預測";
  $("riskReason").textContent = item.riskReason || "-";

  renderModelCompare(item);
  renderFeatures(item);
  renderSimilarCases(item);
}

function renderModelCompare(item) {
  const rows = [
    {
      label: "Mean baseline",
      ratio: item.meanPredictedRemainingRatio,
      rate: item.meanPredictedRemainingRatio == null ? null : 1 - item.meanPredictedRemainingRatio,
    },
    {
      label: "Ridge",
      ratio: item.ridgePredictedRemainingRatio,
      rate: item.ridgePredictedReductionRate,
    },
    {
      label: "Lasso",
      ratio: item.lassoPredictedRemainingRatio,
      rate: item.lassoPredictedReductionRate,
    },
  ];

  $("modelCompare").innerHTML = rows
    .map((row) => {
      const width = row.ratio == null ? 0 : Math.max(2, Math.min(100, row.ratio * 100));
      return `
        <div class="model-row">
          <span class="model-name">${escapeHtml(row.label)}</span>
          <span class="bar-track"><span class="bar-fill" style="width:${width}%"></span></span>
          <span class="model-value">${pct(row.ratio)}</span>
        </div>
      `;
    })
    .join("");

  $("actualCompare").innerHTML = `
    <strong>AI 假設版回測對照</strong><br />
    是否酌減：${yesNo(item.actualIsReduced)}；分類預測：${yesNo(item.predictedIsReduced)}；命中：${yesNo(item.classificationCorrect)}<br />
    實際准許比例：${pct(item.actualRemainingRatio)}；實際酌減率：${pct(item.actualReductionRate)}；區間：${escapeHtml(item.actualBucket || "—")}<br />
    Ridge 誤差：${num(item.ridgeAbsError, 3)}；Lasso 誤差：${num(item.lassoAbsError, 3)}；Mean baseline 誤差：${num(item.meanAbsError, 3)}
  `;
}

function renderFeatures(item) {
  const byModel = state.data.contributionsByJid[item.jid] || {};
  const rows = byModel[state.contributionModel] || [];
  const modelLabel = modelNames[state.contributionModel] || state.contributionModel;

  if (!rows.length) {
    $("featureSummary").textContent = `${modelLabel} 沒有可顯示的特徵貢獻`;
    $("featureList").innerHTML = "";
    return;
  }

  $("featureSummary").innerHTML = `
    <strong>${escapeHtml(modelLabel)}</strong>｜貢獻值來自標準化特徵 × 模型係數
  `;
  $("featureList").innerHTML = rows
    .map((row) => {
      const contribution = Number(row.contribution || 0);
      const cls = contribution >= 0 ? "contrib-positive" : "contrib-negative";
      return `
        <div class="feature-row">
          <span class="feature-label">${escapeHtml(row.label)}</span>
          <span class="feature-number">${num(row.value, 3)}</span>
          <span class="feature-number ${cls}">${num(row.contribution, 3)}</span>
          <span>${escapeHtml(row.interpretation)}</span>
        </div>
      `;
    })
    .join("");
}

function renderSimilarCases(item) {
  const rows = state.data.similarCasesByJid[item.jid] || [];
  $("similarCount").textContent = `${rows.length} 件`;
  if (!rows.length) {
    $("similarCases").innerHTML = `<div class="empty-state">沒有相似案例</div>`;
    return;
  }

  $("similarCases").innerHTML = rows
    .map((row) => {
      const ratioText = row.remainingRatio == null ? "無比例標註" : `准許比例 ${pct(row.remainingRatio)}`;
      return `
        <article class="similar-row">
          <div class="similar-score">${num(row.score, 2)}</div>
          <div>
            <p class="similar-title">#${escapeHtml(row.rank)} ${escapeHtml(row.title || "未命名案件")}</p>
            <p class="similar-meta">${escapeHtml(row.jid)}｜${escapeHtml(row.year)}｜${escapeHtml(row.court)}｜${ratioText}</p>
            <p class="similar-terms">${escapeHtml(row.sharedTerms || "無共同詞")}</p>
            <p class="similar-snippet">${escapeHtml(row.reductionSnippet || row.delaySnippet || "無片段")}</p>
          </div>
        </article>
      `;
    })
    .join("");
}

function setDefaultSelection() {
  const preferred =
    state.data.cases.find((item) => item.split === "test_2025" && item.riskLevel === "高") ||
    state.data.cases.find((item) => item.split === "test_2025") ||
    state.data.cases[0];
  state.selectedJid = preferred?.jid || null;
}

async function bootstrap() {
  try {
    const response = await fetch(DATA_URL);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.data = await response.json();
    $("dataNotice").textContent = `${state.data.metadata.caseCount} 件案件｜${state.data.metadata.notice}`;
    initFilters();
    setDefaultSelection();
    applyFilters();
  } catch (error) {
    $("dataNotice").textContent = `資料載入失敗：${error.message}`;
    $("caseList").innerHTML = `<div class="empty-state">請用本機伺服器開啟此工具</div>`;
  }
}

bootstrap();
