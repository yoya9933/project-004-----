# 工程違約金風險模型治理平台

以工程契約違約金案件為題的資料分析、模型回測與模型治理專案。專案將模型訓練與展示介面分離：模型只能透過離線 pipeline 建立與評估，Streamlit 僅讀取經過 promotion gate 核准的 artifacts，不會在 UI 中重新訓練模型。

> 本專案用於資料分析、模型治理與研究展示，不構成法律意見，也不能取代法院判斷或人工專業審查。

## 核心設計

目前程式碼以 `src/legal_risk_modeling/` 作為模型與資料處理的主要 source of truth。

- `features.py`：特徵定義與特徵資料正規化。
- `models.py`：回歸與分類模型定義，以及統一的 hyperparameter search space。
- `promotion.py`：模型 promotion gate。
- `manifest.py`：run manifest、Git SHA、資料與 artifact SHA256、Python/platform 與套件版本紀錄。
- `classification.py`：是否酌減分類流程。
- `rag_benchmark.py`：相似案例檢索 benchmark。
- `data_quality.py`：資料品質檢查。
- `ui.py`：只讀 approved artifacts 的 Streamlit UI。

根目錄的 `streamlit_app.py` 只負責啟動 UI，不包含 sklearn 訓練邏輯。

## 模型治理流程

### 1. 資料切分

違約金比例模型採年度 out-of-time split：

| Split | 年度 | 用途 |
| --- | --- | --- |
| Train | 2021–2023 | 模型訓練 |
| Validation | 2024 | 模型選擇與 promotion 判斷 |
| Test | 2025 | 獨立回測 |
| Latest | 2026 | 最新年度 regression check |

### 2. Ratio model candidates

目前模型候選由 `src/legal_risk_modeling/models.py` 統一定義：

- Mean baseline
- Ridge Regression
- Lasso
- Elastic Net
- Random Forest
- Extra Trees
- Gradient Boosting
- HistGradientBoosting

每個模型 family 先在 2024 validation split 選出最佳 candidate，再對 Train / Validation / Test / Latest 產生一致的 metrics 與 predictions。

### 3. Promotion gate

learned model 不會因為 validation 指標較好就自動成為正式模型。

目前 gate 會檢查：

- Validation RMSE 至少比 Mean baseline 改善 `0.01`。
- Validation MAE 至少改善 `0.005`。
- 2025 Test RMSE 不可比 baseline 更差。
- 2026 Latest RMSE 不可比 baseline 更差。

若沒有 learned model 通過 gate，`approved_release.json` 會自動將 UI default 回退到 `mean_baseline`。

目前此 branch 的核准結果即採用這個 fallback：尚無 learned ratio model 同時通過所有 gate。

## Streamlit Dashboard

Streamlit 不負責訓練模型，只會讀取：

- `approved_release.json`
- `promotion_report.json`
- `metrics.csv`
- `predictions.csv`
- `manifest.json`（若存在）

UI 提供：

- Model governance 狀態
- Approved model 選擇
- 案件 JID 搜尋
- Actual / Predicted remaining ratio
- 2021–2026 backtest metrics
- Promotion decision 與未通過原因
- Run manifest 與 artifact provenance

UI 只允許 Mean baseline 與列在 `approved_models` 中的模型被展示。

## Generated artifacts

新的 governed runs 預設輸出到：

```text
.artifacts/
├── reduction_ratio_model_expanded_824_sklearn/
├── is_reduced_classification/
├── rag_benchmark/
├── data_quality/
└── benchmark_history/
```

`.artifacts/` 屬於 generated output，不應提交至 Git。

可透過環境變數改變 artifact root：

```text
LEGAL_RISK_ARTIFACT_ROOT=/custom/path
```

若 `.artifacts/reduction_ratio_model_expanded_824_sklearn/` 尚未建立，UI 仍可相容讀取既有的 legacy release：

```text
06_交付物/reduction_ratio_model_expanded_824_sklearn/
```

## 安裝

需要 Python 3.12。

建議使用 lock file 建立可重現環境：

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

## 執行離線 pipeline

### Data quality

```bash
python 04_執行稿/run_data_quality_check.py \
  --input-csv 06_交付物/ai_rag_annotation/annotation_workbook.csv \
  --output-json .artifacts/data_quality/annotation_workbook.json
```

### Build AI / RAG artifacts

```bash
python 04_執行稿/build_ai_rag_annotation_pack.py \
  --output-dir .artifacts/ai_rag_annotation
```

### RAG benchmark

```bash
python 04_執行稿/run_rag_retrieval_benchmark.py \
  --retrieval-csv .artifacts/ai_rag_annotation/rag_similar_cases.csv \
  --metadata-csv 06_交付物/ai_rag_annotation/annotation_workbook.csv \
  --min-queries 20 \
  --min-hit-rate 0.40 \
  --min-mrr 0.28
```

### Governed ratio training

```bash
python 04_執行稿/run_reduction_ratio_model_sklearn.py
```

此步驟會產生：

- `metrics.csv`
- `predictions.csv`
- `selected_models.csv`
- `validation_search.csv`
- `promotion_report.json`
- `approved_release.json`
- `manifest.json`

### Classification pipeline

```bash
python 04_執行稿/run_is_reduced_classification.py \
  --input-csv 06_交付物/ai_rag_annotation/annotation_workbook.csv
```

## 啟動 Dashboard

必須先有可用的 approved release artifacts。

```bash
streamlit run streamlit_app.py
```

預設網址：

```text
http://localhost:8501
```

## 測試與 CI

本專案已有 repository-level automated checks：

```bash
python -m pytest -q
ruff check src/legal_risk_modeling streamlit_app.py tests
```

GitHub Actions 的 Model governance CI 會執行：

- locked dependency install + `pip check`
- Ruff lint
- Python compile check
- unit / architecture / Streamlit tests
- source data quality validation
- RAG benchmark
- governed ratio training
- classification pipeline
- manifest / SHA verification
- Streamlit health smoke test
- governed artifacts retention

另有 Security CI 使用 CodeQL 掃描 Python 程式碼。

## Repository structure

```text
.
├── src/legal_risk_modeling/     # 核心 library
├── tests/                       # governance / architecture / UI tests
├── 04_執行稿/                  # executable pipelines
├── 05_測試與驗證/              # 驗證資料與歷史輸出
├── 06_交付物/                  # source / legacy delivery artifacts
├── .github/workflows/           # CI / Security CI
├── streamlit_app.py             # thin Streamlit entry point
├── pyproject.toml
├── requirements.txt
└── requirements.lock
```

## 目前限制

- 現有資料包含研究用與 AI 輔助標註內容，不能視為法律事實資料庫。
- Ratio model 目前沒有 learned model 通過完整 promotion gate，因此 UI default 為 Mean baseline。
- 模型輸出只代表目前資料切分、特徵與治理政策下的回測結果。
- 任何正式法律或工程決策仍需回到原始判決、契約文件與人工專業判斷。
