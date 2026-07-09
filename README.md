# 工程違約金風險評估工具

## Streamlit 啟動

在本專案根目錄執行：

```powershell
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

開啟：

```text
http://localhost:8501
```

若 8501 port 已被占用，可改用：

```powershell
streamlit run streamlit_app.py --server.port 8518
```

## 專案入口

- Streamlit app：`streamlit_app.py`
- Python 依賴：`requirements.txt`
- 現場訓練輸入：`06_交付物/ai_rag_annotation_expanded_824/annotation_workbook_ai_assumed.csv`
- RAG 相似案例輸入：`06_交付物/ai_rag_annotation_expanded_824/rag_similar_cases.csv`
- Streamlit 驗證腳本：`04_執行稿/validate_streamlit_app.py`

## 現場訓練

Streamlit 開啟後不會自動訓練模型。請在左側側欄按「現場訓練模型」，app 會直接從 `annotation_workbook_ai_assumed.csv` 產生特徵、切分資料、訓練 Logistic / Ridge / Lasso，並把結果暫存在本次 Streamlit session。

## 內容

- 824 件 AI 假設版高相關案件。
- Streamlit 側欄支援搜尋、年度/切分篩選，以及 2025 測試、2026 最新年度展示快捷鍵。
- 按下現場訓練後，每案顯示酌減機率、預測准許比例與預測酌減率。
- 顯示 Ridge、Lasso、mean baseline 的准許比例比較。
- 顯示重要模型特徵與前 5 件 RAG 相似案例。
- 支援下載目前篩選後的案件 JSON。

## 測試

```powershell
python .\04_執行稿\validate_streamlit_app.py
```

驗證結果會輸出到：

- `05_測試與驗證/streamlit_app_validation.json`
- `05_測試與驗證/streamlit_app_validation.md`

## 限制

目前資料為 AI 假設版標註與回測展示，不是法律意見；正式結論須回到原判決全文人工查核。

舊版靜態 HTML/CSS/JS 工具已移除；目前展示入口統一使用根目錄 `streamlit_app.py`。
