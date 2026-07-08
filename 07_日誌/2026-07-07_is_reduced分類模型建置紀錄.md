# 2026-07-07 是否酌減分類模型建置紀錄

## 目標

依專題設計第三階段建立「法院是否酌減違約金」分類模型管線。此階段先完成可重跑的特徵矩陣與回測腳本；正式標籤不足時，腳本必須安全停止，不以 AI 初標或關鍵詞命中替代人工標註。

## 使用資料

- 標註工作表：`06_交付物/ai_rag_annotation/annotation_workbook.csv`
- 待精標案件：120 件
- 目標欄位：`is_reduced`
- 時間切分設定：2021-2024 訓練、2025 驗證、2026 測試

## 實作檔案

- `04_執行稿/run_is_reduced_classification.ps1`

本腳本不依賴 Python、pandas、sklearn 或外部套件，避免目前環境找不到全域 Python 時無法重跑。PowerShell 5 對 UTF-8 無 BOM 的中文腳本較敏感，因此本腳本已轉為 UTF-8 BOM。

## 管線內容

- 讀取 `annotation_workbook.csv`。
- 產出 `feature_matrix.csv`，包含案件基本欄位、正式標籤欄位、金額/天數特徵、人工 issue 欄位、AI 輔助 issue 欄位、候選金額/天數數量與關鍵詞規則特徵。
- 只將正式 `is_reduced` 欄位視為目標標籤。
- 預設不把 `x_has_discretion`、`x_has_mfa_252`、`x_has_over_high` 放入 Logistic Regression；這些欄位只作為關鍵詞規則 baseline 或篩選診斷，降低把判決結論語句當作預測特徵的風險。
- 標籤達門檻後可輸出：
  - majority baseline
  - keyword rule baseline
  - L2 Logistic Regression
  - Accuracy、Precision、Recall、F1-score、ROC-AUC
  - `metrics.csv`、`predictions.csv`、`model_coefficients.csv`

## 產出

| 產物 | 位置 | 筆數/說明 |
|---|---|---:|
| 特徵矩陣 | `06_交付物/is_reduced_classification/feature_matrix.csv` | 120 |
| 已標籤子表 | `06_交付物/is_reduced_classification/labeled_feature_matrix.csv` | 0 |
| 狀態 JSON | `06_交付物/is_reduced_classification/model_status.json` | skipped |
| 狀態說明 | `06_交付物/is_reduced_classification/分類模型狀態.md` | — |

## 驗證

- `feature_matrix.csv`：120 筆。
- `labeled_feature_matrix.csv`：0 筆。
- `model_status.json`：
  - `status` = `skipped`
  - `reason` = `not_enough_official_is_reduced_labels`
  - `labeled_rows` = 0
  - `min_labeled_rows` = 30

## 注意事項

- 目前不能宣稱已完成分類模型訓練，因為正式 `is_reduced` 標籤仍為 0 筆。
- AI/RAG 產生的 `ai_suggest_*` 欄位可以作為模型特徵或人工查核提示，但不能作為 `is_reduced` 目標標籤。
- 若後續人工已補 `claimed_penalty` 與 `allowed_penalty`，可選擇同步補 `is_reduced`；腳本也提供 `-UseDerivedLabelFromAmounts` 參數，但正式報告仍應說明該標籤來源。
- 下一步應優先補標 `is_reduced`，並同步補 `contract_price`、`claimed_penalty`、`delay_days` 與主要 issue 欄位。
