# 工程違約金預測與 RAG 回測專題設計

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

### 2. AI/RAG 輔助標註

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
- AI 提示詞模板：`06_交付物/ai_rag_annotation/ai_annotation_prompt_template.md`。
- 每案提示詞封包：`06_交付物/ai_rag_annotation/case_prompt_packets.jsonl`。
- 人工查核說明：`06_交付物/ai_rag_annotation/human_review_guide.md`。

正式標註欄位仍保持空白，所有 `ai_*_candidates` 與 `ai_suggest_*` 欄位只作為人工查核輔助。

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
- 目前正式 `is_reduced` 標籤為 0 筆，因此模型訓練與回測安全跳過，狀態記錄於 `model_status.json` 與 `分類模型狀態.md`。
- 腳本預設只使用正式人工標註的 `is_reduced` 作為目標，不把 AI 初標、相似案例或關鍵詞命中當成真實標籤。
- 補標達門檻後，腳本會依 2021-2024 訓練、2025 驗證、2026 測試輸出 majority baseline、關鍵詞規則 baseline 與 Logistic Regression 結果。

模型輸入設計上，較可能帶有判決結論的詞，例如 `酌減`、`民法第252條`、`過高`，目前保留於特徵矩陣與關鍵詞規則 baseline，不作為預設 Logistic Regression 特徵。若要做未判決案件風險評估，應優先使用契約金額、主張違約金、逾期天數、展延爭點、業主/承包商可歸責等事實型特徵。

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
- 目前 `claimed_penalty` 與 `allowed_penalty` 正式欄位皆為 0 筆，因此可用比例目標為 0 筆，模型訓練與回測安全跳過。
- 腳本只用經人工確認的 `claimed_penalty` 與 `allowed_penalty` 推導目標，不直接預測原始判賠金額。
- 腳本不把 `allowed_penalty` 放入模型特徵，避免把答案作為輸入。
- 補標達門檻後，腳本會依 2021-2024 訓練、2025 驗證、2026 測試輸出 mean baseline、Ridge Regression 與 Lasso Regression 的 MAE、RMSE、R² 與分段命中率。

目前採用的 `reduction_bucket` 分段規則：

- `未酌減`：`remaining_ratio >= 0.99`。
- `小幅酌減`：`0.70 <= remaining_ratio < 0.99`。
- `中度酌減`：`0.30 <= remaining_ratio < 0.70`。
- `大幅酌減`：`0.05 < remaining_ratio < 0.30`。
- `全免或近乎全免`：`remaining_ratio <= 0.05`。

Random Forest Regressor 與 LightGBM Regressor 可作為後續進階版本，需等 Python 套件環境與標註資料量都穩定後再接入。現階段先保留可解釋、可重跑的比例模型管線。

### 5. 回測設計

採時間切分，模擬用過去案件輔助判斷未來案件。

- 訓練集：2021-2024 年。
- 驗證集：2025 年。
- 測試集：2026 年。

回測時應避免：

- 把法院判決結論、主文或酌減結果當成模型輸入。
- 同案不同審級分散到訓練集與測試集，造成資料洩漏。
- 只報告模型分數，未與 baseline 比較。

### 6. 實務應用：工程違約金風險評估工具

最後成果可設計成一個輔助評估流程，而不是法院判決預測器。

輸入：

- 工程類型。
- 契約總價。
- 主張違約金。
- 逾期天數。
- 是否有展延申請。
- 是否有業主可歸責因素。
- 是否已部分完工或已使用。
- 判決或案件事實文字。

輸出：

- 是否可能涉及違約金酌減風險。
- 可能酌減區間。
- 重要影響特徵。
- 相似判決案例。
- 需人工查核的風險提醒。

## 預期成果

1. 工程違約金案件候選池。
2. 人工/AI 輔助標註資料表。
3. 是否酌減分類模型。
4. 酌減比例迴歸或分段分類模型。
5. 2021-2026 時間切分回測結果。
6. RAG 相似案例檢索與模型解釋流程。
7. 工程違約金風險評估工具原型。

## 報告表述建議

本專題不宣稱能取代法院判斷，也不宣稱能準確預測個案判決結果。較適當的定位是：

> 本研究透過工程判決文本資料、AI 輔助標註與可解釋機器學習模型，探索工程逾期違約金酌減因素與法院處理結果之關聯，並以時間切分回測評估模型作為風險辨識與相似案例檢索工具的可行性。
