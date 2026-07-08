# -*- coding: utf-8 -*-
from __future__ import annotations

import html
import json
import math
from pathlib import Path
from textwrap import dedent
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "06_交付物" / "risk_assessment_tool" / "data" / "risk_tool_data.json"

RISK_RANK = {"高": 3, "中": 2, "低": 1}
RISK_CLASS = {"高": "risk-high", "中": "risk-mid", "低": "risk-low"}
SPLIT_ORDER = {
    "train_2021_2023": 1,
    "validation_2024": 2,
    "test_2025": 3,
    "latest_2026": 4,
}
SPLIT_LABELS = {
    "train_2021_2023": "訓練集 2021-2023",
    "validation_2024": "驗證集 2024",
    "test_2025": "測試集 2025",
    "latest_2026": "最新年度 2026",
}
CONTRIBUTION_MODELS = {
    "logistic_regression_l2": "分類 Logistic",
    "ridge_regression_l2": "比例 Ridge",
    "lasso_regression_l1": "比例 Lasso",
}


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def pct(value: Any, digits: int = 1) -> str:
    if not is_number(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def num(value: Any, digits: int = 3) -> str:
    if not is_number(value):
        return "—"
    return f"{float(value):.{digits}f}"


def yes_no(value: Any) -> str:
    if value == 1:
        return "是"
    if value == 0:
        return "否"
    return "—"


def safe(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def html_block(markup: str) -> str:
    return "\n".join(line.strip() for line in dedent(markup).strip().splitlines())


def render_html(markup: str) -> None:
    body = html_block(markup)
    if hasattr(st, "html"):
        st.html(body)
    else:
        st.markdown(body, unsafe_allow_html=True)


def clamp_ratio(value: Any) -> float:
    if not is_number(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


@st.cache_data(show_spinner=False)
def load_payload() -> dict[str, Any]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到資料檔：{DATA_PATH}")
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    for key in ["metadata", "cases", "similarCasesByJid", "contributionsByJid"]:
        if key not in payload:
            raise KeyError(f"risk_tool_data.json 缺少必要欄位：{key}")
    return payload


def install_style() -> None:
    render_html(
        """
        <style>
          :root {
            --tool-bg: #eef0ef;
            --tool-surface: #ffffff;
            --tool-soft: #f7f8f7;
            --tool-ink: #202124;
            --tool-muted: #646760;
            --tool-line: #cfd5d1;
            --tool-accent: #176b5d;
            --tool-accent-soft: #d9eee8;
            --risk-high: #b64235;
            --risk-mid: #a06a16;
            --risk-low: #2f7658;
          }
          .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
          }
          .tool-title {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            padding: 1rem 1.1rem;
            border: 1px solid var(--tool-line);
            border-radius: 8px;
            background: var(--tool-surface);
            min-width: 0;
            overflow-wrap: anywhere;
          }
          .tool-title h1 {
            margin: 0;
            color: var(--tool-ink);
            font-size: 1.65rem;
            line-height: 1.2;
            letter-spacing: 0;
          }
          .eyebrow {
            margin: 0 0 0.25rem 0;
            color: var(--tool-accent);
            font-size: 0.82rem;
            font-weight: 800;
          }
          .muted {
            color: var(--tool-muted);
            font-size: 0.9rem;
            line-height: 1.6;
            overflow-wrap: anywhere;
          }
          .risk-badge {
            min-width: 5.8rem;
            border-radius: 999px;
            padding: 0.45rem 0.75rem;
            color: #ffffff;
            text-align: center;
            font-weight: 900;
            white-space: nowrap;
          }
          .risk-high { background: var(--risk-high); }
          .risk-mid { background: var(--risk-mid); }
          .risk-low { background: var(--risk-low); }
          .panel {
            border: 1px solid var(--tool-line);
            border-radius: 8px;
            padding: 1rem;
            background: var(--tool-surface);
            min-width: 0;
            overflow-wrap: anywhere;
          }
          .panel h3 {
            margin: 0 0 0.75rem 0;
            font-size: 1.05rem;
            letter-spacing: 0;
          }
          .bar-row {
            display: grid;
            grid-template-columns: minmax(6rem, 8rem) minmax(0, 1fr) minmax(3.8rem, 4.5rem);
            gap: 0.65rem;
            align-items: center;
            margin: 0.55rem 0;
            font-size: 0.93rem;
          }
          .bar-row > span {
            min-width: 0;
            overflow-wrap: anywhere;
          }
          .bar-label {
            font-weight: 800;
          }
          .bar-track {
            height: 0.8rem;
            overflow: hidden;
            border-radius: 999px;
            background: #e8e3d7;
          }
          .bar-fill {
            height: 100%;
            border-radius: inherit;
            background: var(--tool-accent);
          }
          .detail-box {
            margin-top: 0.9rem;
            padding-top: 0.8rem;
            border-top: 1px solid var(--tool-line);
            color: var(--tool-muted);
            font-size: 0.9rem;
            line-height: 1.7;
          }
          .similar-card {
            display: grid;
            grid-template-columns: 4rem minmax(0, 1fr);
            gap: 0.8rem;
            border-top: 1px solid var(--tool-line);
            padding: 0.9rem 0 0.2rem;
            min-width: 0;
          }
          .similar-score {
            display: grid;
            place-items: center;
            width: 3.3rem;
            height: 3.3rem;
            border-radius: 8px;
            background: var(--tool-accent-soft);
            color: var(--tool-accent);
            font-weight: 900;
          }
          .similar-title {
            margin: 0;
            font-weight: 900;
            line-height: 1.35;
          }
          .similar-meta,
          .similar-terms,
          .similar-snippet {
            margin: 0.22rem 0 0;
            font-size: 0.9rem;
            line-height: 1.55;
            overflow-wrap: anywhere;
          }
          .similar-meta,
          .similar-terms {
            color: var(--tool-muted);
          }
          .feature-positive {
            color: var(--risk-high);
            font-weight: 800;
          }
          .feature-negative {
            color: var(--risk-low);
            font-weight: 800;
          }
          .feature-table {
            display: grid;
            gap: 0.35rem;
            margin-top: 0.6rem;
          }
          .feature-grid {
            display: grid;
            grid-template-columns: minmax(9rem, 1.15fr) minmax(4.5rem, 0.45fr) minmax(4.5rem, 0.45fr) minmax(10rem, 1fr);
            gap: 0.65rem;
            align-items: center;
            padding: 0.62rem 0;
            border-top: 1px solid var(--tool-line);
            font-size: 0.9rem;
          }
          .feature-grid > span {
            min-width: 0;
            overflow-wrap: anywhere;
          }
          .feature-header {
            border-top: 0;
            padding-top: 0;
            color: var(--tool-muted);
            font-size: 0.78rem;
            font-weight: 900;
          }
          .feature-label {
            font-weight: 850;
          }
          .feature-number {
            text-align: right;
            font-variant-numeric: tabular-nums;
          }
          .notice {
            border: 1px solid #eadba9;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            background: #fff8e5;
            color: #5c4a20;
            line-height: 1.7;
            overflow-wrap: anywhere;
          }
          @media (max-width: 900px) {
            .feature-grid {
              grid-template-columns: minmax(0, 1fr) minmax(4.5rem, 0.45fr) minmax(4.5rem, 0.45fr);
            }
            .feature-grid .feature-interpretation {
              grid-column: 1 / -1;
            }
          }
          @media (max-width: 700px) {
            .tool-title,
            .bar-row,
            .similar-card {
              display: block;
            }
            .feature-header {
              display: none;
            }
            .feature-grid {
              display: block;
            }
            .feature-grid > span {
              display: block;
              margin: 0.2rem 0;
            }
            .feature-number {
              text-align: left;
            }
            .risk-badge,
            .bar-track,
            .similar-score {
              margin-top: 0.5rem;
            }
          }
        </style>
        """
    )


def ensure_session_defaults() -> None:
    defaults = {
        "search_query": "",
        "year_filter": "全部年度",
        "split_filter": "全部切分",
        "risk_filter": "全部風險",
        "contribution_model": "logistic_regression_l2",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def apply_preset(name: str) -> None:
    st.session_state.search_query = ""
    st.session_state.year_filter = "全部年度"
    st.session_state.split_filter = "全部切分"
    st.session_state.risk_filter = "全部風險"
    if name == "high":
        st.session_state.risk_filter = "高"
    elif name == "test2025":
        st.session_state.year_filter = "2025"
        st.session_state.split_filter = "test_2025"
    elif name == "latest2026":
        st.session_state.year_filter = "2026"
        st.session_state.split_filter = "latest_2026"


def split_label(value: str) -> str:
    if value == "全部切分":
        return value
    return SPLIT_LABELS.get(value, value)


def filter_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyword = str(st.session_state.search_query).strip().lower()
    year = "" if st.session_state.year_filter == "全部年度" else st.session_state.year_filter
    split = "" if st.session_state.split_filter == "全部切分" else st.session_state.split_filter
    risk = "" if st.session_state.risk_filter == "全部風險" else st.session_state.risk_filter

    rows: list[dict[str, Any]] = []
    for item in cases:
        haystack = f"{item.get('jid', '')} {item.get('title', '')} {item.get('court', '')}".lower()
        if keyword and keyword not in haystack:
            continue
        if year and str(item.get("year")) != year:
            continue
        if split and item.get("split") != split:
            continue
        if risk and item.get("riskLevel") != risk:
            continue
        rows.append(item)

    return sorted(
        rows,
        key=lambda item: (
            -RISK_RANK.get(item.get("riskLevel"), 0),
            -int(item.get("year") or 0),
            int(item.get("priority") or 999999),
        ),
    )


def preferred_jid(cases: list[dict[str, Any]]) -> str | None:
    for predicate in [
        lambda item: item.get("split") == "test_2025" and item.get("riskLevel") == "高",
        lambda item: item.get("split") == "test_2025",
        lambda item: True,
    ]:
        match = next((item for item in cases if predicate(item)), None)
        if match:
            return str(match.get("jid"))
    return None


def case_label_map(cases: list[dict[str, Any]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for item in cases:
        jid = str(item.get("jid"))
        labels[jid] = (
            f"{item.get('title') or '未命名案件'}"
            f"｜{item.get('year') or '—'}"
            f"｜{item.get('riskLevel') or '—'}風險"
        )
    return labels


def render_sidebar(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    cases: list[dict[str, Any]] = payload["cases"]
    metadata = payload["metadata"]
    years = sorted({str(item.get("year")) for item in cases if item.get("year")})
    splits = sorted(
        {str(item.get("split")) for item in cases if item.get("split")},
        key=lambda value: SPLIT_ORDER.get(value, 99),
    )

    with st.sidebar:
        st.caption("工程違約金")
        st.title("風險評估工具")
        st.caption(f"{metadata.get('caseCount', len(cases))} 件案件")

        st.markdown("**展示快捷篩選**")
        col1, col2 = st.columns(2)
        col1.button("高風險", use_container_width=True, on_click=apply_preset, args=("high",))
        col2.button("2025 測試", use_container_width=True, on_click=apply_preset, args=("test2025",))
        col3, col4 = st.columns(2)
        col3.button("2026 最新", use_container_width=True, on_click=apply_preset, args=("latest2026",))
        col4.button("清除", use_container_width=True, on_click=apply_preset, args=("clear",))

        st.text_input("搜尋", placeholder="案號、案名、法院", key="search_query")
        st.selectbox("年度", ["全部年度", *years], key="year_filter")
        st.selectbox("切分", ["全部切分", *splits], key="split_filter", format_func=split_label)
        st.selectbox("風險", ["全部風險", "高", "中", "低"], key="risk_filter")

    filtered = filter_cases(cases)
    labels = case_label_map(filtered)

    with st.sidebar:
        st.markdown(f"**符合條件：{len(filtered)} 件**")
        if not filtered:
            st.warning("沒有符合條件的案件")
            return filtered, None

        jids = [str(item.get("jid")) for item in filtered]
        if st.session_state.get("selected_jid") not in jids:
            st.session_state.selected_jid = preferred_jid(filtered)
        selected_jid = st.selectbox(
            "案件",
            jids,
            key="selected_jid",
            format_func=lambda jid: labels.get(jid, jid),
        )

        download_data = json.dumps(filtered, ensure_ascii=False, indent=2)
        st.download_button(
            "下載目前篩選 JSON",
            data=download_data,
            file_name="risk_tool_filtered_cases.json",
            mime="application/json",
            use_container_width=True,
        )

    selected = next((item for item in filtered if str(item.get("jid")) == selected_jid), None)
    return filtered, selected


def render_header(item: dict[str, Any]) -> None:
    risk_level = str(item.get("riskLevel") or "低")
    risk_class = RISK_CLASS.get(risk_level, "risk-low")
    render_html(
        f"""
        <div class="tool-title">
          <div>
            <p class="eyebrow">{safe(item.get("year"))}｜{safe(item.get("splitLabel"))}｜{safe(item.get("court"))}</p>
            <h1>{safe(item.get("title") or "未命名案件")}</h1>
            <p class="muted">{safe(item.get("jid"))}</p>
          </div>
          <div class="risk-badge {risk_class}">{safe(risk_level)}風險</div>
        </div>
        """
    )


def render_metrics(item: dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("酌減機率", pct(item.get("reductionProbability")), help="Logistic Regression")
    cols[1].metric("預測准許比例", pct(item.get("ridgePredictedRemainingRatio")), help="Ridge Regression")
    cols[2].metric("預測酌減率", pct(item.get("ridgePredictedReductionRate")), help="1 - remaining_ratio")
    cols[3].metric("酌減區間", item.get("ridgePredictedBucket") or "無比例預測")
    st.caption(item.get("riskReason") or "—")


def render_model_compare(item: dict[str, Any]) -> None:
    rows = [
        ("Mean baseline", item.get("meanPredictedRemainingRatio")),
        ("Ridge", item.get("ridgePredictedRemainingRatio")),
        ("Lasso", item.get("lassoPredictedRemainingRatio")),
    ]
    parts = ['<div class="panel"><h3>模型比較</h3>']
    for label, ratio in rows:
        width = max(2.0, clamp_ratio(ratio) * 100) if is_number(ratio) else 0.0
        parts.append(
            f"""
            <div class="bar-row">
              <span class="bar-label">{safe(label)}</span>
              <span class="bar-track"><span class="bar-fill" style="width:{width:.1f}%"></span></span>
              <span>{pct(ratio)}</span>
            </div>
            """
        )
    detail = f"""
      <div class="detail-box">
        <strong>AI 假設版回測對照</strong><br>
        是否酌減：{yes_no(item.get("actualIsReduced"))}；
        分類預測：{yes_no(item.get("predictedIsReduced"))}；
        命中：{yes_no(item.get("classificationCorrect"))}<br>
        實際准許比例：{pct(item.get("actualRemainingRatio"))}；
        實際酌減率：{pct(item.get("actualReductionRate"))}；
        區間：{safe(item.get("actualBucket") or "—")}<br>
        Ridge 誤差：{num(item.get("ridgeAbsError"))}；
        Lasso 誤差：{num(item.get("lassoAbsError"))}；
        Mean baseline 誤差：{num(item.get("meanAbsError"))}
      </div>
    """
    parts.append(detail + "</div>")
    render_html("\n".join(parts))


def render_feature_summary(item: dict[str, Any]) -> None:
    render_html(
        f"""
        <div class="panel">
          <h3>模型方向摘要</h3>
          <p class="muted"><strong>提高酌減機率：</strong>{safe(item.get("topClassificationTowardReduction") or "—")}</p>
          <p class="muted"><strong>降低酌減機率：</strong>{safe(item.get("topClassificationTowardNoReduction") or "—")}</p>
          <p class="muted"><strong>Ridge 較少酌減：</strong>{safe(item.get("topRatioTowardLessReductionRidge") or "—")}</p>
          <p class="muted"><strong>Ridge 較多酌減：</strong>{safe(item.get("topRatioTowardMoreReductionRidge") or "—")}</p>
        </div>
        """
    )


def render_features(item: dict[str, Any], contributions: dict[str, Any]) -> None:
    st.subheader("重要特徵")
    model = st.radio(
        "模型切換",
        list(CONTRIBUTION_MODELS.keys()),
        key="contribution_model",
        format_func=lambda value: CONTRIBUTION_MODELS[value],
        horizontal=True,
    )
    rows = contributions.get(str(item.get("jid")), {}).get(model, [])
    if not rows:
        st.info("這個模型沒有可顯示的特徵貢獻。")
        return

    visible_rows = [
        """
        <div class="feature-grid feature-header">
          <span>特徵</span>
          <span class="feature-number">數值</span>
          <span class="feature-number">貢獻</span>
          <span class="feature-interpretation">解讀</span>
        </div>
        """
    ]

    for row in rows[:12]:
        contribution = row.get("contribution")
        contribution_text = num(contribution)
        if is_number(contribution):
            cls = "feature-positive" if float(contribution) >= 0 else "feature-negative"
            contribution_text = f'<span class="{cls}">{contribution_text}</span>'
        visible_rows.append(
            f"""
            <div class="feature-grid">
              <span class="feature-label">{safe(row.get("label"))}</span>
              <span class="feature-number">{num(row.get("value"))}</span>
              <span class="feature-number">{contribution_text}</span>
              <span class="feature-interpretation">{safe(row.get("interpretation"))}</span>
            </div>
            """
        )

    st.markdown("貢獻值來自標準化特徵 × 模型係數，僅解釋目前回測模型方向。")
    st.caption("正負方向依不同模型目標解讀；請搭配人工查核。")
    render_html(f'<div class="feature-table">{"".join(visible_rows)}</div>')


def render_similar_cases(item: dict[str, Any], similar_by_jid: dict[str, Any]) -> None:
    rows = similar_by_jid.get(str(item.get("jid")), [])
    st.subheader(f"RAG 相似案例（{len(rows)} 件）")
    if not rows:
        st.info("沒有相似案例。")
        return

    for row in rows:
        ratio_text = "無比例標註" if not is_number(row.get("remainingRatio")) else f"准許比例 {pct(row.get('remainingRatio'))}"
        snippet = row.get("reductionSnippet") or row.get("delaySnippet") or "無片段"
        render_html(
            f"""
            <div class="similar-card">
              <div class="similar-score">{num(row.get("score"), 2)}</div>
              <div>
                <p class="similar-title">#{safe(row.get("rank"))} {safe(row.get("title") or "未命名案件")}</p>
                <p class="similar-meta">{safe(row.get("jid"))}｜{safe(row.get("year"))}｜{safe(row.get("court"))}｜{safe(ratio_text)}</p>
                <p class="similar-terms">{safe(row.get("sharedTerms") or "無共同詞")}</p>
                <p class="similar-snippet">{safe(snippet)}</p>
              </div>
            </div>
            """
        )


def render_overview(payload: dict[str, Any], filtered: list[dict[str, Any]]) -> None:
    metadata = payload["metadata"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("全部案件", metadata.get("caseCount", len(payload["cases"])))
    col2.metric("目前篩選", len(filtered))
    col3.metric("相似案例", metadata.get("similarCaseCount", "—"))
    col4.metric("特徵貢獻", metadata.get("featureContributionCount", "—"))

    risk_counts = metadata.get("riskCounts", {})
    split_counts = metadata.get("splitCounts", {})
    left, right = st.columns(2)
    with left:
        st.markdown("**風險分布**")
        st.json(risk_counts, expanded=False)
    with right:
        st.markdown("**時間切分**")
        st.json({SPLIT_LABELS.get(k, k): v for k, v in split_counts.items()}, expanded=False)


def render_notice(payload: dict[str, Any]) -> None:
    metadata = payload["metadata"]
    risk_rule = metadata.get("riskRule", {})
    render_html(
        f"""
        <div class="notice">
          <strong>限制說明</strong><br>
          {safe(metadata.get("notice") or "本工具僅供展示與回測解釋。")}<br>
          模型成果應定位為爭點分類、風險辨識或酌減可能性回測，不能宣稱可取代法院判斷或直接預測個案結果。
        </div>
        """
    )
    st.markdown("**風險規則**")
    for level in ["高", "中", "低"]:
        st.write(f"- {level}風險：{risk_rule.get(level, '—')}")


def main() -> None:
    st.set_page_config(
        page_title="工程違約金風險評估工具",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    install_style()
    ensure_session_defaults()

    try:
        payload = load_payload()
    except Exception as exc:  # noqa: BLE001
        st.error(f"資料載入失敗：{exc}")
        st.stop()

    filtered, selected = render_sidebar(payload)
    if not selected:
        st.title("工程違約金風險評估工具")
        st.info("請調整左側篩選條件。")
        render_notice(payload)
        return

    render_header(selected)
    render_metrics(selected)

    tab_case, tab_data, tab_limits = st.tabs(["案件儀表板", "資料總覽", "限制說明"])
    with tab_case:
        left, right = st.columns([0.95, 1.05])
        with left:
            render_model_compare(selected)
        with right:
            render_feature_summary(selected)
        render_features(selected, payload["contributionsByJid"])
        render_similar_cases(selected, payload["similarCasesByJid"])

    with tab_data:
        render_overview(payload, filtered)
        rows = [
            {
                "案號": item.get("jid"),
                "案名": item.get("title"),
                "年度": item.get("year"),
                "切分": item.get("splitLabel"),
                "法院": item.get("court"),
                "風險": item.get("riskLevel"),
                "酌減機率": pct(item.get("reductionProbability")),
                "預測准許比例": pct(item.get("ridgePredictedRemainingRatio")),
            }
            for item in filtered
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    with tab_limits:
        render_notice(payload)
        st.markdown("**人工查核重點**")
        st.write("- 確認金額是否為契約總價、主張違約金、法院准許違約金或其他款項。")
        st.write("- 確認法院最終結論、展延、歸責、損害與使用收益等爭點。")
        st.write("- RAG 相似案例只作閱讀優先順序，不直接套用法律結論。")


if __name__ == "__main__":
    main()
