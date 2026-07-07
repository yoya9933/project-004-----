# AI/RAG 輔助標註人工查核說明

## 建議流程

1. 先開啟 `annotation_workbook.csv`，依 `annotation_priority` 順序處理。
2. 參考 `source_snippet_penalty`、`source_snippet_delay`、`source_snippet_reduction` 找到判決相關段落。
3. 參考 `ai_*_candidates`，但不得直接複製為正式標註。
4. 回到 `json_file` 指向的原判決全文，確認金額角色與法院最後判斷。
5. 填入正式欄位：`contract_price`、`delay_days`、`claimed_penalty`、`allowed_penalty`、`is_reduced`、`remaining_ratio`、`reduction_rate`、爭點欄位與 `key_reason`。
6. 使用 `rag_similar_cases.csv` 查看相似案例，輔助理解常見酌減理由，但不要把相似案例結論套用到本案。

## 必查事項

- 金額是否為契約總價、請求違約金、法院准許違約金，還是利息/訴訟費/保留款/工程款。
- 法院是否真的酌減，或只是引用當事人請求酌減的主張。
- 判決是否為上訴審、發回、部分廢棄，避免把前審或當事人主張當成最終結果。
- 同案不同審級若要進模型，後續回測切分需避免資料洩漏。

## 欄位填寫

- `is_reduced`：0/1。若法院准許違約金小於主張違約金，填 1；否則填 0。無法確認留空。
- `remaining_ratio`：`allowed_penalty / claimed_penalty`。
- `reduction_rate`：`1 - remaining_ratio`。
- 爭點欄位：0/1。只在法院理由或重要事實中明確涉及時填 1。
