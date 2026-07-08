# 120 件最後判決金額抽取摘要

本表以 `annotation_workbook.csv` 的 120 件案件為清單，回讀原始 JSON 判決全文，優先抽取 `主文` 中法院命令給付、返還、支付或賠償的本金金額；若上訴審主文寫成「命給付逾/超過某金額部分廢棄」，則另以保留額標示；若主文明確駁回且無給付，暫記為 0。

## 抽取結果

- 案件數：120
- 有數值結果：108
- `appeal_dismissed_no_main_amount`：12
- `appeal_retained_amount`：13
- `first_instance_dismissed_no_award`：19
- `main_text_award`：76

## 重要限制

- `final_judgment_amount` 是本次從判決主文或末段結論抽出的「本案判決命令給付/返還金額」，不等於既有模型欄位 `allowed_penalty`。
- `principal_award_components` 會列出主文直接命令給付、返還、支付或賠償的本金組成；多筆本金時，`final_judgment_amount` 為組成加總。
- `appeal_retained_amount` 代表上訴審主文寫成「命給付逾/超過某金額部分廢棄」，本表暫以未被廢棄的保留額作為最後金額。
- 若狀態為 `appeal_dismissed_no_main_amount`，代表該判決主文僅寫上訴或抗告駁回，未明列維持的前審金額，需要回前審判決確認。
- 返還支票、定存單或履約保證文件且主文記載票面/面額時，本表會保留該金額作為候選本金組成；正式研究使用前仍建議人工確認法律意義。

## 建議優先查核案件

| JID | 年度 | 案名 | 狀態 | 目前金額 |
|---|---:|---|---|---:|
| TNHV,109,建上更一,3,20210420,1 | 2021 | 給付工程款 | `appeal_dismissed_no_main_amount` |  |
| KSHV,109,建上,33,20220316,1 | 2022 | 給付工程款 | `appeal_dismissed_no_main_amount` |  |
| HLHV,109,建上,6,20220413,1 | 2022 | 給付違約金等 | `appeal_retained_amount` | 585221 |
| TCHV,108,建上更一,25,20220614,1 | 2022 | 給付工程款等 | `appeal_retained_amount` | 3033583 |
| TNHV,109,建上,22,20221011,1 | 2022 | 延遲給付工程款之利息等 | `appeal_retained_amount` | 220619 |
| TPHV,110,建上,9,20221230,1 | 2022 | 給付工程款 | `appeal_retained_amount` | 49428591 |
| TPHV,109,建上,50,20230117,1 | 2023 | 給付工程款 | `appeal_retained_amount` | 29704395 |
| TNHV,111,上更一,30,20230509,1 | 2023 | 履行契約 | `appeal_retained_amount` | 3844947 |
| HLHV,111,建上更二,3,20230616,1 | 2023 | 給付工程款等 | `appeal_dismissed_no_main_amount` |  |
| TCHV,106,建上,51,20230614,1 | 2023 | 給付工程款 | `appeal_dismissed_no_main_amount` |  |
| KSHV,109,建上,10,20230823,1 | 2023 | 給付工程款 | `appeal_retained_amount` | 10643259 |
| TPHV,109,建上,7,20230913,2 | 2023 | 給付工程款等 | `appeal_dismissed_no_main_amount` |  |
| TPHV,111,建上,33,20231212,1 | 2023 | 給付工程款 | `appeal_retained_amount` | 312776 |
| TNHV,111,重上,118,20240424,1 | 2024 | 債務不履行等 | `appeal_dismissed_no_main_amount` |  |
| TPHV,111,建上,61,20240416,1 | 2024 | 給付工程款等 | `appeal_retained_amount` | 640128 |
| TCHV,112,建上,55,20240918,1 | 2024 | 變更設計展延工期管理費及損害賠償等 | `appeal_dismissed_no_main_amount` |  |
| TCHV,111,建上,84,20241210,1 | 2024 | 損害賠償等 | `appeal_retained_amount` | 2793577 |
| TNHV,112,建上,25,20250213,1 | 2025 | 給付工程款 | `appeal_dismissed_no_main_amount` |  |
| KSHV,112,建上,5,20250226,2 | 2025 | 給付工程款 | `appeal_retained_amount` | 410533 |
| TPHV,112,重上更一,108,20250319,1 | 2025 | 給付工程款等 | `appeal_retained_amount` | 1108878 |
| TNHV,113,建上,14,20250508,1 | 2025 | 給付工程款 | `appeal_retained_amount` | 1771698 |
| TPHV,113,建上,46,20250610,1 | 2025 | 返還違約金 | `appeal_dismissed_no_main_amount` |  |
| TNHV,110,建上,18,20251226,1 | 2025 | 給付工程款 | `appeal_dismissed_no_main_amount` |  |
| KSHV,114,上,88,20251217,1 | 2025 | 給付承攬報酬 | `appeal_dismissed_no_main_amount` |  |
| TPHV,113,重上,4,20260429,1 | 2026 | 履行契約 | `appeal_dismissed_no_main_amount` |  |
