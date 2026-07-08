# 檢索增強生成相似案例檢索與模型解釋流程

## 目的

本流程把既有 RAG 相似案例檢索結果、`is_reduced` 分類模型、`remaining_ratio` 比例模型與模型係數整合成案件級解釋包。它的定位是輔助人工閱讀、風險辨識與報告展示，不是自動判決或法律意見。

## 一次重跑命令

```powershell
python .\04_執行稿\build_rag_model_explanation_pack.py
```

## 輸入

- 標註資料：`06_交付物/ai_rag_annotation/annotation_workbook.csv`
- RAG 索引：`06_交付物/ai_rag_annotation/rag_case_index.csv`
- 相似案例：`06_交付物/ai_rag_annotation/rag_similar_cases.csv`
- 分類模型：`06_交付物/is_reduced_classification/`
- 比例模型：`06_交付物/reduction_ratio_model/`

## 輸出

- `case_explanation_summary.csv`：每案一列，整合模型預測、特徵貢獻摘要、前 5 件相似案例與查核提示。
- `similar_case_evidence.csv`：每組 query/similar 一列，保留 600 列相似案例證據，並附上 query 的模型預測。
- `model_feature_contributions.csv`：每案、每模型、每特徵的標準化值、係數與貢獻值，可用來追溯模型解釋。
- `case_explanation_cards.md`：120 件案件的可讀式解釋卡。
- `rag_model_explanation_workflow.md`：本流程說明。
- `explanation_status.json`：機器可讀的輸出狀態與列數。

## 解釋流程

1. 先用 `case_explanation_summary.csv` 找到目標案件，確認其時間切分、實際標註、分類模型酌減機率與比例模型預測。
2. 查看 `top_classification_toward_reduction` 與 `top_classification_toward_no_reduction`，判斷分類模型主要被哪些事實型特徵推動。
3. 查看 Ridge 比例模型的 `top_ratio_toward_more_reduction_ridge` 與 `top_ratio_toward_less_reduction_ridge`，判斷模型預測准許比例時的主要方向。
4. 進入 `similar_case_evidence.csv`，閱讀前 5 件相似案例的共同詞、相似度、法院、年度與片段。
5. 回到原判決全文查核金額角色、法院最終結論、展延/歸責/損害等爭點；相似案例不得直接套用結論。
6. 報告時同時呈現 baseline 與模型結果，避免只挑模型分數或只挑相似案例支持單一結論。

## 目前狀態

- 案件解釋列數：120
- 相似案例證據列數：600
- 特徵貢獻列數：6840
- 主要分類模型：`logistic_regression_l2`
- 主要比例模型：`ridge_regression_l2`，並保留 `lasso_regression_l1` 與 `mean_baseline` 比較。

## 限制

- 目前正式欄位為 AI 假設版標註，尚未等同逐案人工查核。
- RAG 相似度主要來自詞彙與爭點重疊，代表閱讀優先順序，不代表法律上案情完全相同。
- 線性模型的特徵貢獻是目前資料與模型下的統計方向，不應被解讀為法院必然採納的法律因果。
- 比例模型在 2025 測試集尚未優於 mean baseline，因此比例預測更適合作為流程展示與後續改良基準。
