from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATION_CSV = (
    PROJECT_ROOT / "06_交付物" / "ai_rag_annotation" / "annotation_workbook.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "06_交付物" / "final_judgment_amounts"

CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "○": 0,
    "一": 1,
    "壹": 1,
    "二": 2,
    "貳": 2,
    "貮": 2,
    "贰": 2,
    "兩": 2,
    "俩": 2,
    "三": 3,
    "參": 3,
    "叁": 3,
    "叄": 3,
    "四": 4,
    "肆": 4,
    "五": 5,
    "伍": 5,
    "六": 6,
    "陸": 6,
    "陆": 6,
    "七": 7,
    "柒": 7,
    "八": 8,
    "捌": 8,
    "九": 9,
    "玖": 9,
}
SMALL_UNITS = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000, "仟": 1000}
CHINESE_NUMBER_CHARS = "".join(CHINESE_DIGITS) + "".join(SMALL_UNITS) + "萬万億亿"
AMOUNT_TOKEN_PATTERN = rf"[0-9{CHINESE_NUMBER_CHARS}][0-9,，{CHINESE_NUMBER_CHARS}]*"
AMOUNT_WITH_UNIT_PATTERN = re.compile(
    rf"(?:新\s*[臺台]\s*幣|臺\s*幣|台\s*幣|NT\$)?"
    rf"\s*(?:[（(]\s*下\s*同\s*[）)])?\s*"
    rf"(?P<amount>{AMOUNT_TOKEN_PATTERN})\s*(?:元|圓)"
)
MAIN_END_PATTERN = re.compile(
    r"事\s*實\s*及\s*理\s*由|事\s*實|理\s*由|犯\s*罪\s*事\s*實|中\s*華\s*民\s*國"
)
AWARD_CONTEXT_PATTERN = re.compile(
    r"(?:被告|原告|上訴人|被上訴人|反訴被告|反訴原告|再審原告|再審被告|"
    r"抗告人|相對人|第三人|債務人|債權人).{0,80}?"
    r"(?:應(?:再|連帶|共同|各)?(?:給付|返還|支付|賠償|給與|清償)|"
    r"應負給付|給付|返還|支付|賠償)"
)
DIRECT_AWARD_PATTERN = re.compile(
    r"應.{0,8}?(?:給付|返還|支付|賠償|給與|清償)|"
    r"應負給付|應再給付|應連帶給付|應共同給付"
)
EXCLUDED_CONTEXT_PATTERN = re.compile(
    r"訴訟費用|裁判費|假執行|供擔保|預供擔保|免為假執行|擔保金|執行費"
)
FIRST_INSTANCE_DISMISS_PATTERN = re.compile(
    r"(?:原告|反訴原告|聲請人|上訴人|被上訴人).{0,20}(?:之訴|聲請).{0,20}駁回|"
    r"(?:原告|反訴原告|上訴人|被上訴人).{0,20}請求.{0,20}(?:無理由|不應准許)"
)
APPEAL_DISMISS_PATTERN = re.compile(r"(?:上訴|附帶上訴|抗告|再抗告|再審之訴).{0,30}駁回")
CONCLUSION_CUES = ["綜上所述", "從而", "是以", "爰判決", "應予准許", "為有理由"]


@dataclass
class AmountCandidate:
    source: str
    amount_text: str
    amount: int
    sentence: str
    is_award_context: bool
    is_excluded_context: bool
    is_retained_context: bool


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_judgment_text(text: str) -> str:
    value = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u3000", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"(?<=[0-9,，])\s+(?=[0-9,，])", "", value)
    value = re.sub(r"(?<=[0-9,，])\s+(?=[萬万億亿千仟百佰十拾元圓])", "", value)
    value = re.sub(r"(?<=[萬万億亿千仟百佰十拾])\s+(?=[0-9" + CHINESE_NUMBER_CHARS + r"])", "", value)
    value = re.sub(r"(?<=[" + CHINESE_NUMBER_CHARS + r"])\s+(?=[" + CHINESE_NUMBER_CHARS + r"])", "", value)
    value = re.sub(r"(?<=[新臺台幣])\s+(?=[0-9" + CHINESE_NUMBER_CHARS + r"（(])", "", value)
    value = re.sub(r"(?<=[應再連帶共同各給付返還支付賠償給與清償負])\s+(?=[應再連帶共同各給付返還支付賠償給與清償負])", "", value)
    return value


def compact(text: Any, limit: int = 220) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def normalize_amount_token(token: str) -> str:
    return (
        token.replace(",", "")
        .replace("，", "")
        .replace(" ", "")
        .replace("万", "萬")
        .replace("亿", "億")
    )


def parse_section_amount(section: str) -> int | None:
    text = normalize_amount_token(section)
    if not text:
        return None
    if text.isdigit():
        return int(text)

    total = 0
    current: int | None = None
    digit_buffer = ""
    found = False

    for char in text:
        if char.isdigit():
            digit_buffer += char
            found = True
            continue
        if digit_buffer:
            current = int(digit_buffer)
            digit_buffer = ""
        if char in CHINESE_DIGITS:
            current = CHINESE_DIGITS[char]
            found = True
            continue
        if char in SMALL_UNITS:
            number = current if current not in (None, 0) else 1
            total += number * SMALL_UNITS[char]
            current = None
            found = True
            continue
        return None

    if digit_buffer:
        current = int(digit_buffer)
    if current is not None:
        total += current
    return total if found else None


def parse_amount(token: str) -> int | None:
    text = normalize_amount_token(token)
    if not text:
        return None
    if text.isdigit():
        return int(text)

    total = 0
    remainder = text
    if "億" in remainder:
        head, remainder = remainder.split("億", 1)
        section = parse_section_amount(head or "1")
        if section is None:
            return None
        total += section * 100_000_000
    if "萬" in remainder:
        head, remainder = remainder.split("萬", 1)
        section = parse_section_amount(head or "1")
        if section is None:
            return None
        total += section * 10_000
    if remainder:
        section = parse_section_amount(remainder)
        if section is None:
            return None
        total += section
    return total


def split_statement_units(block: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", block)
    cleaned = re.sub(
        r"\s*(?=(?:[一二三四五六七八九十壹貳參肆伍陸柒捌玖拾]+、|[0-9]+、|[㈠㈡㈢㈣㈤㈥㈦㈧㈨㈩]))",
        "。",
        cleaned,
    )
    units = re.split(r"(?<=[。；;])", cleaned)
    return [unit.strip() for unit in units if unit.strip()]


def extract_main_block(full_text: str) -> tuple[str, str]:
    text = normalize_judgment_text(full_text)
    main_match = re.search(r"主\s*文", text)
    if not main_match:
        return "", text

    start = main_match.end()
    rest = text[start:]
    end_match = MAIN_END_PATTERN.search(rest)
    if end_match and end_match.start() >= 8:
        return rest[: end_match.start()].strip(), text
    return rest[:2500].strip(), text


def extract_conclusion_block(text: str) -> str:
    windows: list[str] = []
    for cue in CONCLUSION_CUES:
        start = 0
        while True:
            idx = text.find(cue, start)
            if idx < 0:
                break
            windows.append(text[max(0, idx - 300) : idx + 1300])
            start = idx + len(cue)
    windows.append(text[-3000:])
    return "\n".join(windows)


def extract_amount_candidates(block: str, source: str) -> list[AmountCandidate]:
    candidates: list[AmountCandidate] = []
    for sentence in split_statement_units(block):
        amount_matches = list(AMOUNT_WITH_UNIT_PATTERN.finditer(sentence))
        if not amount_matches:
            continue
        for match in amount_matches:
            amount_text = match.group("amount")
            amount = parse_amount(amount_text)
            if amount is None:
                continue
            award_context = sentence[max(0, match.start() - 320) : match.end() + 120]
            local_context = sentence[max(0, match.start() - 120) : match.end() + 120]
            before = sentence[max(0, match.start() - 160) : match.start()]
            after = sentence[match.end() : match.end() + 100]
            is_retained = is_retained_award_amount(before, after)
            is_award = bool(AWARD_CONTEXT_PATTERN.search(award_context)) or bool(
                DIRECT_AWARD_PATTERN.search(award_context)
            )
            is_excluded = bool(EXCLUDED_CONTEXT_PATTERN.search(local_context)) or is_retained
            candidates.append(
                AmountCandidate(
                    source=source,
                    amount_text=amount_text,
                    amount=amount,
                    sentence=sentence,
                    is_award_context=is_award,
                    is_excluded_context=is_excluded,
                    is_retained_context=is_retained,
                )
            )
    return candidates


def is_retained_award_amount(before: str, after: str) -> bool:
    if re.search(r"利息部分.{0,80}$", before) or re.search(r"利息部分", after[:40]):
        return False
    if re.search(
        r"(?:命|判命|關於).{0,80}(?:給付|返還|支付|賠償).{0,20}"
        r"(?:逾|超過|超出)\s*(?:「|『)?\s*(?:新\s*[臺台]\s*幣|臺\s*幣|台\s*幣|下同)?\s*$",
        before,
    ):
        return True
    if re.search(
        r"(?:給付|返還|支付|賠償).{0,20}"
        r"(?:逾|超過|超出)\s*(?:「|『)?\s*(?:新\s*[臺台]\s*幣|臺\s*幣|台\s*幣|下同)?\s*$",
        before,
    ):
        return True
    return False


def select_candidate(candidates: list[AmountCandidate]) -> AmountCandidate | None:
    preferred = [
        candidate
        for candidate in candidates
        if candidate.is_award_context and not candidate.is_excluded_context
    ]
    if preferred:
        return preferred[0]
    return None


def looks_like_award_continuation(sentence: str) -> bool:
    return bool(
        re.match(
            r"(?:原告|被告|上訴人|被上訴人|反訴原告|反訴被告|再審原告|再審被告|"
            r"[\u4e00-\u9fffA-Za-z0-9（）()]{2,60}"
            r"(?:股份有限公司|有限公司|工程處|管理人|政府|大學|學校|機關|公司))",
            sentence.strip(),
        )
    )


def unique_candidates(candidates: list[AmountCandidate]) -> list[AmountCandidate]:
    seen: set[tuple[int, str]] = set()
    unique: list[AmountCandidate] = []
    for candidate in candidates:
        key = (candidate.amount, candidate.sentence)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def extract_principal_award_components(block: str) -> list[AmountCandidate]:
    components: list[AmountCandidate] = []
    carry_award = False
    for sentence in split_statement_units(block):
        sentence_candidates = extract_amount_candidates(sentence, "main_text")
        eligible = [
            candidate
            for candidate in sentence_candidates
            if candidate.is_award_context
            and not candidate.is_excluded_context
            and not candidate.is_retained_context
        ]
        if eligible:
            components.append(eligible[0])
            carry_award = True
            continue

        continuation_candidates = [
            candidate
            for candidate in sentence_candidates
            if not candidate.is_excluded_context and not candidate.is_retained_context
        ]
        if carry_award and continuation_candidates and looks_like_award_continuation(sentence):
            components.append(continuation_candidates[0])
            continue

        if "駁回" in sentence or "訴訟費用" in sentence or "假執行" in sentence:
            carry_award = False
    return unique_candidates(components)


def extract_retained_award_components(candidates: list[AmountCandidate]) -> list[AmountCandidate]:
    return unique_candidates([candidate for candidate in candidates if candidate.is_retained_context])


def component_text(candidates: list[AmountCandidate]) -> str:
    return " ; ".join(f"{candidate.amount_text}={candidate.amount}" for candidate in candidates)


def component_snippet(candidates: list[AmountCandidate], limit: int = 380) -> str:
    return " || ".join(compact(candidate.sentence, limit) for candidate in candidates)


def parse_number(value: Any) -> int | None:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def load_judgment(row: dict[str, str]) -> dict[str, Any]:
    json_file = Path(row.get("json_file", ""))
    if not json_file.is_absolute():
        json_file = PROJECT_ROOT / json_file
    with json_file.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def amount_list(candidates: list[AmountCandidate], limit: int = 12) -> str:
    parts = [f"{candidate.amount_text}={candidate.amount}" for candidate in candidates[:limit]]
    if len(candidates) > limit:
        parts.append(f"...(+{len(candidates) - limit})")
    return " ; ".join(parts)


def extraction_review_note(status: str, selected: AmountCandidate | None) -> str:
    if status == "main_text_award":
        return "主文直接命令給付/返還之本金金額；多筆本金時已加總，仍建議人工確認。"
    if status == "appeal_retained_amount":
        return "主文以「逾/超過某金額部分廢棄」表示保留金額；本欄暫以保留額作為最後判決金額，需人工確認前審脈絡。"
    if status == "conclusion_award":
        return "主文未抓到可用金額，改由末段結論抓取；需回主文人工確認。"
    if status == "first_instance_dismissed_no_award":
        return "主文顯示一審/本訴請求駁回且未命令給付，暫記為 0。"
    if status == "appeal_dismissed_no_main_amount":
        return "主文為上訴/抗告駁回但未明列金額；需查前審主文才能得知維持金額。"
    if selected and selected.is_excluded_context:
        return "抽到的候選金額位於假執行、擔保或訴訟費用脈絡，不作為最後判決金額。"
    return "未能自動抽出最後判決金額；需人工讀取主文。"


def extract_row(row: dict[str, str]) -> dict[str, Any]:
    judgment = load_judgment(row)
    full_text = judgment.get("JFULL", "")
    main_block, normalized_text = extract_main_block(full_text)
    conclusion_block = extract_conclusion_block(normalized_text)

    main_candidates = extract_amount_candidates(main_block, "main_text")
    conclusion_candidates = extract_amount_candidates(conclusion_block, "conclusion")
    principal_components = extract_principal_award_components(main_block)
    retained_components = extract_retained_award_components(main_candidates)

    selected = select_candidate(main_candidates)
    status = "main_text_award" if selected else ""

    if selected is None:
        if retained_components:
            selected = retained_components[0]
            status = "appeal_retained_amount"
        elif FIRST_INSTANCE_DISMISS_PATTERN.search(main_block) and not re.search(
            r"應(?:再|連帶|共同|各)?(?:給付|返還|支付|賠償|給與|清償)", main_block
        ):
            status = "first_instance_dismissed_no_award"
        elif APPEAL_DISMISS_PATTERN.search(main_block):
            status = "appeal_dismissed_no_main_amount"
        else:
            selected = select_candidate(conclusion_candidates)
            status = "conclusion_award" if selected else ""

    main_hint = compact(main_block, 320)
    if selected is None:
        if status == "first_instance_dismissed_no_award":
            amount_value: int | None = 0
            amount_text = "0"
            snippet = main_hint
        elif status == "appeal_dismissed_no_main_amount":
            amount_value = None
            amount_text = ""
            snippet = main_hint
        else:
            status = "not_found"
            amount_value = None
            amount_text = ""
            snippet = main_hint
    else:
        amount_value = selected.amount
        amount_text = selected.amount_text
        snippet = compact(selected.sentence, 320)

    if status == "main_text_award" and principal_components:
        amount_value = sum(candidate.amount for candidate in principal_components)
        amount_text = " + ".join(candidate.amount_text for candidate in principal_components)
        snippet = component_snippet(principal_components, 220)
    elif status == "appeal_retained_amount" and retained_components:
        amount_value = sum(candidate.amount for candidate in retained_components)
        amount_text = " + ".join(candidate.amount_text for candidate in retained_components)
        snippet = component_snippet(retained_components, 220)

    ai_allowed = parse_number(row.get("allowed_penalty"))
    difference = ""
    ratio_to_ai_allowed = ""
    if amount_value is not None and ai_allowed not in (None, 0):
        difference = amount_value - ai_allowed
        ratio_to_ai_allowed = round(amount_value / ai_allowed, 6)

    return {
        "JID": row.get("JID", judgment.get("JID", "")),
        "decision_year": row.get("decision_year", judgment.get("JYEAR", "")),
        "month": row.get("month", ""),
        "court": row.get("court", ""),
        "JTITLE": row.get("JTITLE", judgment.get("JTITLE", "")),
        "JCASE": row.get("JCASE", judgment.get("JCASE", "")),
        "JNO": row.get("JNO", judgment.get("JNO", "")),
        "JDATE": row.get("JDATE", judgment.get("JDATE", "")),
        "json_file": row.get("json_file", ""),
        "final_judgment_amount": amount_value if amount_value is not None else "",
        "final_judgment_amount_text": amount_text,
        "extraction_status": status,
        "principal_award_total": sum(candidate.amount for candidate in principal_components)
        if principal_components
        else "",
        "principal_award_components": component_text(principal_components),
        "retained_award_total": sum(candidate.amount for candidate in retained_components)
        if retained_components
        else "",
        "retained_award_components": component_text(retained_components),
        "source_sentence": snippet,
        "main_text_excerpt": main_hint,
        "main_amount_candidates": amount_list(main_candidates),
        "conclusion_amount_candidates": amount_list(conclusion_candidates),
        "ai_allowed_penalty": row.get("allowed_penalty", ""),
        "ai_claimed_penalty": row.get("claimed_penalty", ""),
        "final_minus_ai_allowed_penalty": difference,
        "final_to_ai_allowed_penalty_ratio": ratio_to_ai_allowed,
        "manual_checked": row.get("manual_checked", ""),
        "review_note": extraction_review_note(status, selected),
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    counts = Counter(str(row["extraction_status"]) for row in rows)
    amount_rows = [row for row in rows if str(row.get("final_judgment_amount", "")).strip()]
    review_rows = [
        row
        for row in rows
        if row["extraction_status"]
        not in {"main_text_award", "first_instance_dismissed_no_award"}
    ]

    lines = [
        "# 120 件最後判決金額抽取摘要",
        "",
        "本表以 `annotation_workbook.csv` 的 120 件案件為清單，回讀原始 JSON 判決全文，優先抽取 `主文` 中法院命令給付、返還、支付或賠償的本金金額；若上訴審主文寫成「命給付逾/超過某金額部分廢棄」，則另以保留額標示；若主文明確駁回且無給付，暫記為 0。",
        "",
        "## 抽取結果",
        "",
        f"- 案件數：{len(rows)}",
        f"- 有數值結果：{len(amount_rows)}",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"- `{status}`：{count}")

    lines.extend(
        [
            "",
            "## 重要限制",
            "",
            "- `final_judgment_amount` 是本次從判決主文或末段結論抽出的「本案判決命令給付/返還金額」，不等於既有模型欄位 `allowed_penalty`。",
            "- `principal_award_components` 會列出主文直接命令給付、返還、支付或賠償的本金組成；多筆本金時，`final_judgment_amount` 為組成加總。",
            "- `appeal_retained_amount` 代表上訴審主文寫成「命給付逾/超過某金額部分廢棄」，本表暫以未被廢棄的保留額作為最後金額。",
            "- 若狀態為 `appeal_dismissed_no_main_amount`，代表該判決主文僅寫上訴或抗告駁回，未明列維持的前審金額，需要回前審判決確認。",
            "- 返還支票、定存單或履約保證文件且主文記載票面/面額時，本表會保留該金額作為候選本金組成；正式研究使用前仍建議人工確認法律意義。",
            "",
            "## 建議優先查核案件",
            "",
        ]
    )
    if review_rows:
        lines.append("| JID | 年度 | 案名 | 狀態 | 目前金額 |")
        lines.append("|---|---:|---|---|---:|")
        for row in review_rows[:30]:
            lines.append(
                f"| {row['JID']} | {row['decision_year']} | {row['JTITLE']} | "
                f"`{row['extraction_status']}` | {row.get('final_judgment_amount', '')} |"
            )
    else:
        lines.append("無。")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract final judgment amounts for the 120 annotated cases.")
    parser.add_argument("--annotation-csv", type=Path, default=DEFAULT_ANNOTATION_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = read_csv(args.annotation_csv)
    output_rows = [extract_row(row) for row in rows]

    fieldnames = [
        "JID",
        "decision_year",
        "month",
        "court",
        "JTITLE",
        "JCASE",
        "JNO",
        "JDATE",
        "json_file",
        "final_judgment_amount",
        "final_judgment_amount_text",
        "extraction_status",
        "principal_award_total",
        "principal_award_components",
        "retained_award_total",
        "retained_award_components",
        "source_sentence",
        "main_text_excerpt",
        "main_amount_candidates",
        "conclusion_amount_candidates",
        "ai_allowed_penalty",
        "ai_claimed_penalty",
        "final_minus_ai_allowed_penalty",
        "final_to_ai_allowed_penalty_ratio",
        "manual_checked",
        "review_note",
    ]
    output_csv = args.output_dir / "final_judgment_amounts.csv"
    summary_md = args.output_dir / "final_judgment_amounts_summary.md"
    write_csv(output_csv, output_rows, fieldnames)
    write_summary(summary_md, output_rows)

    counts = Counter(row["extraction_status"] for row in output_rows)
    print(f"Wrote {len(output_rows)} rows to {output_csv}")
    print(f"Wrote summary to {summary_md}")
    print(dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
