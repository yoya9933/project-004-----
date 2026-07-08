from __future__ import annotations

import argparse
import json
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = PROJECT_ROOT / "06_交付物" / "risk_assessment_tool"
OUTPUT_DIR = PROJECT_ROOT / "05_測試與驗證"


class AssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stylesheets: list[str] = []
        self.scripts: list[str] = []
        self.visible_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "link" and attr.get("rel") == "stylesheet" and attr.get("href"):
            self.stylesheets.append(attr["href"] or "")
        if tag == "script" and attr.get("src"):
            self.scripts.append(attr["src"] or "")

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.visible_text.append(text)


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_http(base_url: str, paths: list[str], failures: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for relative_path in paths:
        url = base_url.rstrip("/") + "/" + relative_path.lstrip("/")
        try:
            with urllib.request.urlopen(url, timeout=8) as response:
                content = response.read()
                status = response.status
        except Exception as exc:  # noqa: BLE001
            failures.append(f"HTTP check failed for {url}: {exc}")
            results.append({"url": url, "status": "error", "length": 0})
            continue
        results.append({"url": url, "status": status, "length": len(content)})
        require(status == 200, f"HTTP status is not 200 for {url}: {status}", failures)
        require(len(content) > 0, f"HTTP content is empty for {url}", failures)
    return results


def validate(base_url: str | None = None) -> dict[str, Any]:
    failures: list[str] = []
    index_path = TOOL_DIR / "index.html"
    style_path = TOOL_DIR / "assets" / "styles.css"
    script_path = TOOL_DIR / "assets" / "app.js"
    data_path = TOOL_DIR / "data" / "risk_tool_data.json"
    status_path = TOOL_DIR / "data" / "risk_tool_status.json"

    for path in [index_path, style_path, script_path, data_path, status_path]:
        require(path.exists(), f"Missing file: {path}", failures)
        if path.exists():
            require(path.stat().st_size > 0, f"Empty file: {path}", failures)

    parser = AssetParser()
    if index_path.exists():
        parser.feed(index_path.read_text(encoding="utf-8"))
        require("assets/styles.css" in parser.stylesheets, "index.html does not reference assets/styles.css", failures)
        require("assets/app.js" in parser.scripts, "index.html does not reference assets/app.js", failures)
        for required_text in ["酌減機率", "預測准許比例", "預測酌減率", "酌減區間", "重要特徵", "RAG 相似案例"]:
            require(
                any(required_text in text for text in parser.visible_text),
                f"index.html missing visible label: {required_text}",
                failures,
            )

    payload = load_json(data_path) if data_path.exists() else {}
    cases = payload.get("cases", [])
    similar_by_jid = payload.get("similarCasesByJid", {})
    contributions_by_jid = payload.get("contributionsByJid", {})
    metadata = payload.get("metadata", {})

    require(metadata.get("caseCount") == 120, f"metadata.caseCount != 120: {metadata.get('caseCount')}", failures)
    require(len(cases) == 120, f"cases length != 120: {len(cases)}", failures)
    require(metadata.get("similarCaseCount") == 600, f"similarCaseCount != 600: {metadata.get('similarCaseCount')}", failures)
    require(metadata.get("featureContributionCount") == 6840, f"featureContributionCount != 6840: {metadata.get('featureContributionCount')}", failures)
    require(metadata.get("missingReductionProbability") == 0, "missingReductionProbability is not 0", failures)
    require(metadata.get("missingRidgePrediction") == 5, "missingRidgePrediction is not 5", failures)

    required_case_fields = [
        "jid",
        "title",
        "year",
        "split",
        "riskLevel",
        "reductionProbability",
        "ridgePredictedRemainingRatio",
        "ridgePredictedReductionRate",
        "ridgePredictedBucket",
    ]
    missing_probability = 0
    missing_ridge = 0
    for case in cases:
        for field in required_case_fields:
            require(field in case, f"Case missing field {field}: {case.get('jid')}", failures)
        jid = case.get("jid")
        require(len(similar_by_jid.get(jid, [])) == 5, f"Case does not have 5 similar cases: {jid}", failures)
        models = set(contributions_by_jid.get(jid, {}).keys())
        require(
            {"logistic_regression_l2", "ridge_regression_l2", "lasso_regression_l1"}.issubset(models),
            f"Case missing contribution model: {jid}",
            failures,
        )
        if case.get("reductionProbability") is None:
            missing_probability += 1
        if case.get("ridgePredictedRemainingRatio") is None:
            missing_ridge += 1
    require(missing_probability == 0, f"Case reductionProbability missing count != 0: {missing_probability}", failures)
    require(missing_ridge == 5, f"Case ridge prediction missing count != 5: {missing_ridge}", failures)

    http_results: list[dict[str, Any]] = []
    if base_url:
        http_results = check_http(
            base_url,
            ["", "assets/styles.css", "assets/app.js", "data/risk_tool_data.json"],
            failures,
        )

    status = {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "checks": {
            "caseCount": len(cases),
            "similarCaseCount": sum(len(items) for items in similar_by_jid.values()),
            "featureContributionCount": metadata.get("featureContributionCount"),
            "missingReductionProbability": missing_probability,
            "missingRidgePrediction": missing_ridge,
            "htmlStylesheets": parser.stylesheets,
            "htmlScripts": parser.scripts,
            "http": http_results,
        },
    }
    return status


def write_report(status: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "risk_assessment_tool_validation.json"
    md_path = OUTPUT_DIR / "risk_assessment_tool_validation.md"
    json_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 工程違約金風險評估工具驗證",
        "",
        f"- 狀態：{status['status']}",
        f"- 案件數：{status['checks']['caseCount']}",
        f"- 相似案例數：{status['checks']['similarCaseCount']}",
        f"- 特徵貢獻數：{status['checks']['featureContributionCount']}",
        f"- 酌減機率缺值：{status['checks']['missingReductionProbability']}",
        f"- Ridge 比例預測缺值：{status['checks']['missingRidgePrediction']}",
        "",
        "## HTTP 檢查",
        "",
    ]
    if status["checks"]["http"]:
        for item in status["checks"]["http"]:
            lines.append(f"- `{item['url']}`：{item['status']}，{item['length']} bytes")
    else:
        lines.append("- 未提供 `--base-url`，略過 HTTP 檢查。")
    if status["failures"]:
        lines.extend(["", "## 失敗項目", ""])
        lines.extend(f"- {failure}" for failure in status["failures"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the risk assessment tool prototype.")
    parser.add_argument("--base-url", default="", help="Optional local HTTP server base URL.")
    args = parser.parse_args()
    status = validate(args.base_url or None)
    write_report(status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    raise SystemExit(0 if status["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
