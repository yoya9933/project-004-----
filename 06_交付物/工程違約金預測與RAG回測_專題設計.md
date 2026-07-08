# 工程違約金預測與檢索增強生成回測專題設計

## 專題題目

RAG 輔助之工程逾期違約金酌減預測與回測：以 2021-2026 年工程判決資料為基礎

## 核心研究問題

法院在工程逾期違約金案件中，是否會因工期展延、業主可歸責、承包商可歸責、實際損害不明、工程已部分完成或已使用、違約金占契約總價比例等因素，而酌減違約金？能否利用判決文本特徵與人工/AI 輔助標註，建立可回測的輔助風險辨識模型？

## 研究流程

### 1. 關鍵詞篩選

從完整工程判決資料庫中建立候選池。

- 母體資料：2021-2026 年工程相關判決，共 5,906 件 JSON 判決全文。
- 第一層篩選：含 `違約金`、`逾期違約金`、`逾期`、`工期`、`展延` 等詞。
- 第二層篩選：含 `酌減`、`民法第252條`、`過高`、`相當` 等違約金酌減語彙。
- 輸出：逾期違約金候選池、違約金酌減高度相關池、抽樣精標清單。

目前已完成第一版輸出：

- 逾期違約金候選池：`06_交付物/keyword_screening/overdue_penalty_candidates.csv`，1,912 件。
- 違約金酌減高度相關池：`06_交付物/keyword_screening/penalty_reduction_high_relevance.csv`，824 件。
- 抽樣精標清單：`06_交付物/keyword_screening/penalty_reduction_annotation_sample.csv`，120 件。

實作時將 `相當` 保留為輔助命中詞，但不讓只命中 `相當` 的案件單獨進入高度相關池，以降低第二層候選池雜訊。

### 2. 人工智慧與檢索增強生成輔助標註

使用 AI 與 RAG 降低人工閱讀成本，但不直接取代人工判斷。

- RAG 用途一：根據案號或全文找出相似工程違約金案例。
- RAG 用途二：輔助整理法院常見酌減理由與相似案例依據。
- AI 標註用途：初步萃取契約總價、主張違約金、法院准許違約金、逾期天數、展延爭點、法院核心理由。
- 人工查核：所有金額、比例、法院結論與關鍵爭點必須回到原判決全文確認。

建議標註欄位：

- `contract_price`：契約總價。
- `delay_days`：法院認定逾期天數。
- `claimed_penalty`：業主主張或扣罰違約金。
- `allowed_penalty`：法院最後准許違約金。
- `is_reduced`：法院是否酌減。
- `remaining_ratio`：法院准許違約金 / 主張違約金。
- `reduction_rate`：1 - remaining_ratio。
- `issue_owner_fault`：是否涉及業主可歸責。
- `issue_contractor_fault`：是否涉及承包商可歸責。
- `issue_extension_request`：是否涉及展延申請。
- `issue_actual_damage_unclear`：是否涉及實際損害不明。
- `issue_partial_completion`：是否涉及部分完工。
- `issue_used_by_owner`：是否涉及業主已使用工程成果。
- `key_reason`：法院酌減或不酌減的核心理由。

目前已完成第一版標註包：

- 標註工作表：`06_交付物/ai_rag_annotation/annotation_workbook.csv`，120 件。
- RAG 候選索引：`06_交付物/ai_rag_annotation/rag_case_index.csv`，824 件。
- 相似案例表：`06_交付物/ai_rag_annotation/rag_similar_cases.csv`，600 列，每件精標案件最多 5 件相似案例。
- 常見理由摘要：`06_交付物/ai_rag_annotation/reason_pattern_summary.csv`，6 類理由語彙統計。
- AI 提示詞模板：`06_交付物/ai_rag_annotation/人工智慧輔助標註提示詞模板.md`。
- 每案提示詞封包：`06_交付物/ai_rag_annotation/case_prompt_packets.jsonl`。
- 人工查核說明：`06_交付物/ai_rag_annotation/人工智慧與檢索增強生成輔助標註人工查核說明.md`。

原始正式標註欄位曾保持空白，所有 `ai_*_candidates` 與 `ai_suggest_*` 欄位原則上只作為人工查核輔助。

補充進度：依使用者指示，已先假設 AI 判斷正確，將 AI 候選與建議轉入正式欄位並覆寫 `annotation_workbook.csv`。覆寫前原始版已備份為 `annotation_workbook.before_ai_assumed_review_20260707_233448.csv`。目前 `manual_checked` 以 `ai_assumed` 標示，適合用於流程展示與初步回測，但不能等同逐案回到判決全文的嚴格人工查核。

補充進度：已完成 RAG 相似案例與模型解釋整合包，將 824 件 RAG 索引、600 列相似案例、分類/比例模型預測與模型係數整合為案件級解釋流程。輸出位置為 `06_交付物/rag_model_explanation/`。

### 3. 分類模型：是否酌減

第一個主要模型任務為分類問題。

- 目標變數：`is_reduced`
- 問題定義：法院是否將主張違約金酌減？
- 模型建議：
  - Baseline：多數類別、關鍵詞規則。
  - 可解釋模型：Logistic Regression、Linear SVM。
  - 進階模型：Random Forest、LightGBM。
  - 文字模型：TF-IDF + 線性模型，或 embedding + MLP。
- 評估指標：
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC

目前已完成第一版分類管線：

- 腳本：`04_執行稿/run_is_reduced_classification.ps1`。
- 輸出：`06_交付物/is_reduced_classification/`。
- 已產出 `feature_matrix.csv`，共 120 件待精標案件。
- 目前 AI 假設版 `is_reduced` 標籤為 120 筆，其中 `is_reduced=1` 為 74 筆、`is_reduced=0` 為 46 筆。
- 腳本預設只使用正式人工標註的 `is_reduced` 作為目標，不把 AI 初標、相似案例或關鍵詞命中當成真實標籤。
- 已依 2021-2023 訓練、2024 驗證、2025 測試、2026 最新年度外部檢查輸出 majority baseline、關鍵詞規則 baseline 與 Logistic Regression 結果。

模型輸入設計上，較可能帶有判決結論的詞，例如 `酌減`、`民法第252條`、`過高`，目前保留於特徵矩陣與關鍵詞規則 baseline，不作為預設 Logistic Regression 特徵。若要做未判決案件風險評估，應優先使用契約金額、主張違約金、逾期天數、展延爭點、業主/承包商可歸責等事實型特徵。

AI 假設版回測已產出於 `06_交付物/is_reduced_classification/`，另保留同結果副本於 `06_交付物/is_reduced_classification_ai_assumed/`。目前正式測試集改為 2025 年 27 件；Logistic Regression 的 Accuracy 為 0.593、F1-score 為 0.718、ROC-AUC 為 0.641。majority baseline 與關鍵詞規則 baseline 在 2025 測試集的 Accuracy 同為 0.630、F1-score 同為 0.773、ROC-AUC 同為 0.500。2026 年僅 5 件，保留為最新年度外部檢查；Logistic Regression 的 Accuracy 為 0.400、F1-score 為 0.571、ROC-AUC 為 0.500，不作為主要測試結論。

### 4. 迴歸模型：酌減比例

第二個主要模型任務為迴歸或分段分類。

- 目標變數一：`remaining_ratio = allowed_penalty / claimed_penalty`
- 目標變數二：`reduction_rate = 1 - remaining_ratio`
- 可選目標變數三：`reduction_bucket`
  - 未酌減
  - 小幅酌減
  - 中度酌減
  - 大幅酌減
  - 全免或近乎全免

模型建議：

- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- LightGBM Regressor

評估指標：

- MAE
- RMSE
- R²
- 分段命中率

注意：不建議一開始直接預測原始金額，因為工程案件金額差距極大，模型容易被大型案件牽動。比例與區間比較適合回測，也比較能表現法院支持程度。

目前已完成第一版酌減比例模型管線：

- 腳本：`04_執行稿/run_reduction_ratio_model.ps1`。
- 輸出：`06_交付物/reduction_ratio_model/`。
- 已產出 `ratio_model_frame.csv`，共 120 件待精標案件。
- 目前 AI 假設版 `claimed_penalty` 為 120 筆、`allowed_penalty` 為 117 筆，其中 115 筆可建立有效 `remaining_ratio`。
- 腳本只用經人工確認的 `claimed_penalty` 與 `allowed_penalty` 推導目標，不直接預測原始判賠金額。
- 腳本不把 `allowed_penalty` 放入模型特徵，避免把答案作為輸入。
- 已依 2021-2023 訓練、2024 驗證、2025 測試、2026 最新年度外部檢查輸出 mean baseline、Ridge Regression 與 Lasso Regression 的 MAE、RMSE、R² 與分段命中率。

目前採用的 `reduction_bucket` 分段規則：

- `未酌減`：`remaining_ratio >= 0.99`。
- `小幅酌減`：`0.70 <= remaining_ratio < 0.99`。
- `中度酌減`：`0.30 <= remaining_ratio < 0.70`。
- `大幅酌減`：`0.05 < remaining_ratio < 0.30`。
- `全免或近乎全免`：`remaining_ratio <= 0.05`。

Random Forest Regressor 與 LightGBM Regressor 可作為後續進階版本，需等 Python 套件環境與標註資料量都穩定後再接入。現階段先保留可解釋、可重跑的比例模型管線。

AI 假設版比例回測已產出於 `06_交付物/reduction_ratio_model/`，另保留同結果副本於 `06_交付物/reduction_ratio_model_ai_assumed/`。目前正式測試集改為 2025 年 27 件；mean baseline 的 MAE 為 0.3109、RMSE 為 0.3552、R² 為 -0.0030、分段命中率為 0.148。Ridge Regression 的 MAE 為 0.3532、RMSE 為 0.4203、R² 為 -0.4044、分段命中率為 0.037；Lasso Regression 的 MAE 為 0.3557、RMSE 為 0.4134、R² 為 -0.3584、分段命中率為 0.000。這表示目前線性比例模型在 2025 測試集尚未優於簡單平均 baseline，應定位為流程驗證與特徵檢查，而非已具穩定預測力的模型。2026 年 5 件保留為最新年度外部檢查，不作為主要測試結論。

### 5. 回測設計

採時間切分，模擬用過去案件輔助判斷未來案件。

- 訓練集：2021-2023 年。
- 驗證集：2024 年。
- 測試集：2025 年。
- 最新年度外部檢查 / 補充案例：2026 年。

調整原因：目前 2026 年樣本僅 5 件，若作為唯一測試集，分數容易被單一案件大幅牽動。改以 2025 年作為正式測試集後，分類模型測試集有 27 件、比例模型測試集有 27 件，較適合作為主要回測指標；2026 年則保留為最新年度案例的外部檢查與展示補充。

回測時應避免：

- 把法院判決結論、主文或酌減結果當成模型輸入。
- 同案不同審級分散到訓練集與測試集，造成資料洩漏。
- 只報告模型分數，未與 baseline 比較。

### 6. 檢索增強生成相似案例檢索與模型解釋流程

本階段已把 RAG 相似案例檢索與模型解釋整合成可重跑流程。

- 腳本：`04_執行稿/build_rag_model_explanation_pack.py`。
- 輸出資料夾：`06_交付物/rag_model_explanation/`。
- 案件級總表：`case_explanation_summary.csv`，120 件，每案整合模型預測、特徵貢獻摘要與前 5 件相似案例。
- 相似案例證據表：`similar_case_evidence.csv`，600 列，保留 query/similar 關係、相似度、共同詞、片段與 query 模型預測。
- 特徵貢獻表：`model_feature_contributions.csv`，6,840 列，包含每案、每模型、每特徵的標準化值、係數與貢獻值。
- 解釋卡：`case_explanation_cards.md`，120 件可讀式案件解釋卡。
- 流程說明：`rag_model_explanation_workflow.md`，說明如何從案件總表、相似案例與模型特徵貢獻進行人工查核。

解釋流程建議：

1. 先在 `case_explanation_summary.csv` 找到目標案件，確認時間切分、實際標註、分類模型酌減機率與比例模型預測。
2. 查看 `top_classification_toward_reduction`、`top_classification_toward_no_reduction`、`top_ratio_toward_more_reduction_ridge` 與 `top_ratio_toward_less_reduction_ridge`，理解模型主要依據哪些特徵產生方向。
3. 進入 `similar_case_evidence.csv` 查看前 5 件相似案例，閱讀共同詞、相似度、法院、年度與相關片段。
4. 回到原判決全文查核金額角色、法院結論、展延、歸責、損害與使用收益等爭點。
5. 在報告中同時呈現模型預測、baseline、特徵貢獻與相似案例，不把任何單一輸出當成法院結論。

驗證結果：

- 120 件案件皆有案件級解釋列。
- 120 件案件皆有 5 件 RAG 相似案例。
- 分類模型預測無缺值。
- 比例模型有 115 件可用比例目標；其餘 5 件因原始比例目標不可用而未產生 Ridge/Lasso 預測。

限制：RAG 相似度主要代表詞彙與爭點相近，不代表法律上案件相同；模型特徵貢獻只解釋目前模型的統計方向，不代表法院真正採納的法律因果。所有正式結論仍須回到原判決全文人工查核。

### 7. 實務應用：工程違約金風險評估工具

最後成果可設計成一個輔助評估流程，而不是法院判決預測器。

目前已完成 Streamlit 版工具，並保留原靜態版資料包：

- 資料打包腳本：`04_執行稿/build_risk_assessment_tool_data.py`。
- 工具入口：`streamlit_app.py`。
- 資料位置：`06_交付物/risk_assessment_tool/data/risk_tool_data.json`。
- 啟動方式：在專案根目錄執行 `streamlit run streamlit_app.py`，再開啟 `http://localhost:8501`。
- 支援範圍：展示 120 件既有案件，不支援新案件手動輸入。
- 展示主軸：數值預測展示，並保留模型比較、重要特徵、RAG 相似案例與人工查核提醒。

輸入：

- 既有 120 件案件的案號、案名、法院與年度。
- 既有模型輸出：酌減機率、預測准許比例、預測酌減率、酌減區間。
- 既有解釋資料：重要特徵貢獻、RAG 相似案例與人工查核提醒。

輸出：

- 風險等級：高、中、低。
- 酌減機率：Logistic Regression 數值。
- 預測准許比例與預測酌減率：Ridge Regression 數值。
- 酌減區間：未酌減、小幅酌減、中度酌減、大幅酌減或全免/近乎全免。
- baseline 比較：Ridge、Lasso 與 mean baseline 的准許比例。
- 重要影響特徵：分類模型與比例模型的主要特徵貢獻。
- 相似判決案例：每案前 5 件 RAG 相似案例。
- 需人工查核的風險提醒。

工具原型限制：目前只展示 AI 假設版標註與回測結果，不應被描述為可直接預測法院判決；新案件手動輸入與即時計算 RAG 相似案例可列為後續版本。

## 預期成果

1. 工程違約金案件候選池。
2. 人工/AI 輔助標註資料表。
3. 是否酌減分類模型。
4. 酌減比例迴歸或分段分類模型。
5. 2021-2025 時間切分回測結果，並以 2026 作為最新年度外部檢查。
6. RAG 相似案例檢索與模型解釋流程。
7. 工程違約金風險評估工具原型。

## 報告表述建議

本專題不宣稱能取代法院判斷，也不宣稱能準確預測個案判決結果。較適當的定位是：

> 本研究透過工程判決文本資料、AI 輔助標註與可解釋機器學習模型，探索工程逾期違約金酌減因素與法院處理結果之關聯，並以時間切分回測評估模型作為風險辨識與相似案例檢索工具的可行性。
