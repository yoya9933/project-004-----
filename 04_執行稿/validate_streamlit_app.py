from __future__ import annotations

import importlib.util
import json
import py_compile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "streamlit_app.py"
README_PATH = PROJECT_ROOT / "README.md"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
DATA_PATH = PROJECT_ROOT / "06_交付物" / "risk_assessment_tool" / "data" / "risk_tool_data.json"
OUTPUT_DIR = PROJECT_ROOT / "05_測試與驗證"


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate() -> dict[str, Any]:
    failures: list[str] = []

    require(APP_PATH.exists(), f"Missing root Streamlit app: {APP_PATH}", failures)
    require(README_PATH.exists(), f"Missing root README: {README_PATH}", failures)
    require(REQUIREMENTS_PATH.exists(), f"Missing requirements file: {REQUIREMENTS_PATH}", failures)
    require(DATA_PATH.exists(), f"Missing risk tool data: {DATA_PATH}", failures)

    compile_ok = False
    if APP_PATH.exists():
        try:
            py_compile.compile(str(APP_PATH), doraise=True)
            compile_ok = True
        except py_compile.PyCompileError as exc:
            failures.append(f"streamlit_app.py does not compile: {exc.msg}")

    source = APP_PATH.read_text(encoding="utf-8") if APP_PATH.exists() else ""
    for required_text in [
        "streamlit as st",
        "DATA_PATH",
        "render_sidebar",
        "案件儀表板",
        "資料總覽",
        "限制說明",
        "RAG 相似案例",
        "重要特徵",
    ]:
        require(required_text in source, f"streamlit_app.py missing text: {required_text}", failures)

    requirements = REQUIREMENTS_PATH.read_text(encoding="utf-8") if REQUIREMENTS_PATH.exists() else ""
    require("streamlit" in requirements.lower(), "requirements.txt does not include streamlit", failures)

    readme = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""
    for required_text in ["streamlit run streamlit_app.py", "requirements.txt", "risk_tool_data.json"]:
        require(required_text in readme, f"README.md missing text: {required_text}", failures)

    payload = load_json(DATA_PATH) if DATA_PATH.exists() else {}
    cases = payload.get("cases", [])
    similar_by_jid = payload.get("similarCasesByJid", {})
    contributions_by_jid = payload.get("contributionsByJid", {})
    metadata = payload.get("metadata", {})

    require(metadata.get("caseCount") == 120, f"metadata.caseCount != 120: {metadata.get('caseCount')}", failures)
    require(len(cases) == 120, f"cases length != 120: {len(cases)}", failures)
    require(metadata.get("similarCaseCount") == 600, f"similarCaseCount != 600: {metadata.get('similarCaseCount')}", failures)
    require(
        metadata.get("featureContributionCount") == 6840,
        f"featureContributionCount != 6840: {metadata.get('featureContributionCount')}",
        failures,
    )

    missing_similar = 0
    missing_models = 0
    required_models = {"logistic_regression_l2", "ridge_regression_l2", "lasso_regression_l1"}
    for item in cases:
        jid = item.get("jid")
        if len(similar_by_jid.get(jid, [])) != 5:
            missing_similar += 1
        if not required_models.issubset(set(contributions_by_jid.get(jid, {}).keys())):
            missing_models += 1
    require(missing_similar == 0, f"Cases missing 5 similar cases: {missing_similar}", failures)
    require(missing_models == 0, f"Cases missing contribution models: {missing_models}", failures)

    streamlit_available = importlib.util.find_spec("streamlit") is not None

    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "checks": {
            "appPath": str(APP_PATH.relative_to(PROJECT_ROOT)),
            "readmePath": str(README_PATH.relative_to(PROJECT_ROOT)),
            "compileOk": compile_ok,
            "streamlitInstalled": streamlit_available,
            "caseCount": len(cases),
            "similarCaseCount": sum(len(items) for items in similar_by_jid.values()),
            "featureContributionCount": metadata.get("featureContributionCount"),
            "missingSimilar": missing_similar,
            "missingContributionModels": missing_models,
        },
    }


def write_report(status: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "streamlit_app_validation.json"
    md_path = OUTPUT_DIR / "streamlit_app_validation.md"
    json_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Streamlit 風險評估工具驗證",
        "",
        f"- 狀態：{status['status']}",
        f"- App 位置：`{status['checks']['appPath']}`",
        f"- README 位置：`{status['checks']['readmePath']}`",
        f"- Python 編譯：{status['checks']['compileOk']}",
        f"- 本機已安裝 Streamlit：{status['checks']['streamlitInstalled']}",
        f"- 案件數：{status['checks']['caseCount']}",
        f"- 相似案例數：{status['checks']['similarCaseCount']}",
        f"- 特徵貢獻數：{status['checks']['featureContributionCount']}",
        f"- 缺相似案例案件數：{status['checks']['missingSimilar']}",
        f"- 缺特徵模型案件數：{status['checks']['missingContributionModels']}",
    ]
    if status["failures"]:
        lines.extend(["", "## 失敗項目", ""])
        lines.extend(f"- {failure}" for failure in status["failures"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    status = validate()
    write_report(status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    raise SystemExit(0 if status["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
