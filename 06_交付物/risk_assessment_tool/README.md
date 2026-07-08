# 工程違約金風險評估工具原型

## 啟動

```powershell
cd .\06_交付物\risk_assessment_tool
python -m http.server 8000
```

開啟：

```text
http://localhost:8000
```

## 資料重建

```powershell
cd ..\..
python .\04_執行稿\build_risk_assessment_tool_data.py
```

## 內容

- 120 件既有案件。
- 支援搜尋、年度/切分/風險篩選，以及高風險、2025 測試、2026 最新年度展示快捷鍵。
- 每案顯示酌減機率、預測准許比例、預測酌減率、酌減區間與風險等級。
- 顯示 Ridge、Lasso、mean baseline 的准許比例比較。
- 顯示重要模型特徵與前 5 件 RAG 相似案例。

## 測試

```powershell
cd ..\..
python .\04_執行稿\validate_risk_assessment_tool.py --base-url http://127.0.0.1:8000/
```

驗證結果會輸出到：

- `05_測試與驗證/risk_assessment_tool_validation.json`
- `05_測試與驗證/risk_assessment_tool_validation.md`

## 限制

目前資料為 AI 假設版標註與回測展示，不是法律意見；正式結論須回到原判決全文人工查核。
