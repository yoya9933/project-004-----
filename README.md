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
- 工具資料：`06_交付物/risk_assessment_tool/data/risk_tool_data.json`
- 資料打包腳本：`04_執行稿/build_risk_assessment_tool_data.py`
- Streamlit 驗證腳本：`04_執行稿/validate_streamlit_app.py`

## 資料重建

```powershell
python .\04_執行稿\build_risk_assessment_tool_data.py
```

## 內容

- 120 件既有案件。
- Streamlit 側欄支援搜尋、年度/切分/風險篩選，以及高風險、2025 測試、2026 最新年度展示快捷鍵。
- 每案顯示酌減機率、預測准許比例、預測酌減率、酌減區間與風險等級。
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

原靜態 HTML/CSS/JS 檔案仍保留於 `06_交付物/risk_assessment_tool/`，作為舊版展示與資料包結構參考。
