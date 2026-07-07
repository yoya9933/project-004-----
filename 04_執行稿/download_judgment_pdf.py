from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

BASE_URL = "http://140.116.245.86:8090"
DEFAULT_QUERY = "請求給付工程款等"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CSV_NAME = PROJECT_ROOT / "02_輸入資料" / "2021-2026判決總表.csv"
OUTPUT_DIR = PROJECT_ROOT / "02_輸入資料" / "pdf_output"


@dataclass(frozen=True)
class JudgmentRecord:
    month: str
    court: str
    jid: str
    jyear: str
    jcase: str
    jno: str
    jdate: str
    jtitle: str
    jfull_len: str
    engine_hits: str
    hit_basis: str
    jpdf: str
    src_path: str

    @property
    def json_path(self) -> str:
        return self.src_path.strip()

    @property
    def json_url(self) -> str:
        encoded = urllib.parse.quote(self.json_path.replace(os.sep, "/"), safe="/,.")
        return f"{BASE_URL}/{encoded}"


def read_records(csv_path: Path) -> list[JudgmentRecord]:
    records: list[JudgmentRecord] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                JudgmentRecord(
                    month=row.get("month", "").strip(),
                    court=row.get("court", "").strip(),
                    jid=row.get("JID", "").strip(),
                    jyear=row.get("JYEAR", "").strip(),
                    jcase=row.get("JCASE", "").strip(),
                    jno=row.get("JNO", "").strip(),
                    jdate=row.get("JDATE", "").strip(),
                    jtitle=row.get("JTITLE", "").strip(),
                    jfull_len=row.get("JFULL_len", "").strip(),
                    engine_hits=row.get("工程次數", "").strip(),
                    hit_basis=row.get("命中依據", "").strip(),
                    jpdf=row.get("JPDF", "").strip(),
                    src_path=row.get("src_path", "").strip(),
                )
            )
    return records


def find_records(records: Iterable[JudgmentRecord], keyword: str) -> list[JudgmentRecord]:
    keyword = keyword.strip()
    if not keyword:
        return []
    exact = [record for record in records if record.jtitle == keyword]
    if exact:
        return exact
    return [record for record in records if keyword in record.jtitle]


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def find_font_file() -> Path:
    candidates = [
        Path(r"C:\Windows\Fonts\msjh.ttc"),
        Path(r"C:\Windows\Fonts\msjhl.ttc"),
        Path(r"C:\Windows\Fonts\msjhbd.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\mingliu.ttc"),
        Path(r"C:\Windows\Fonts\NotoSansCJKtc-Regular.otf"),
        Path(r"C:\Windows\Fonts\NotoSansCJKtc-Regular.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return path

    fonts_dir = Path(r"C:\Windows\Fonts")
    if fonts_dir.exists():
        for pattern in ("*jhenghei*", "*mingliu*", "*simsun*", "*simhei*", "*notosanscjk*"):
            matches = sorted(fonts_dir.glob(pattern))
            if matches:
                return matches[0]
    raise FileNotFoundError("找不到可用的中文字型，請安裝 Windows 中文字型或指定 --font")


def register_font(font_path: Path) -> str:
    font_name = "JudgmentCJK"
    suffix = font_path.suffix.lower()
    if suffix not in {".ttf", ".ttc", ".otf"}:
        raise ValueError(f"不支援的字型格式: {font_path}")
    pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
    return font_name


def make_paragraph_style(font_name: str, size: int = 11, leading: int | None = None) -> ParagraphStyle:
    return ParagraphStyle(
        name=f"{font_name}-{size}",
        fontName=font_name,
        fontSize=size,
        leading=leading or int(size * 1.45),
        alignment=TA_LEFT,
        spaceAfter=2 * mm,
        wordWrap="CJK",
    )


def safe_text(value: object) -> str:
    return html.escape(str(value or ""))


def split_body(text: str) -> list[str]:
    raw_lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in raw_lines:
        if not line:
            if buffer:
                paragraphs.append("<br/>".join(safe_text(part) for part in buffer))
                buffer = []
            continue
        buffer.append(line)
    if buffer:
        paragraphs.append("<br/>".join(safe_text(part) for part in buffer))
    return paragraphs


def build_pdf(output_path: Path, meta: JudgmentRecord, payload: dict, font_name: str) -> None:
    title_style = make_paragraph_style(font_name, size=16, leading=22)
    meta_style = make_paragraph_style(font_name, size=10, leading=14)
    body_style = make_paragraph_style(font_name, size=11, leading=18)
    body_style.firstLineIndent = 0

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=payload.get("JTITLE", meta.jtitle),
        author="Copilot",
    )

    story = []
    story.append(Paragraph(safe_text(payload.get("JTITLE") or meta.jtitle), title_style))
    story.append(Spacer(1, 4 * mm))

    info_rows = [
        ["案號", safe_text(payload.get("JID") or meta.jid)],
        ["日期", safe_text(payload.get("JDATE") or meta.jdate)],
        ["法院", safe_text(meta.court)],
        ["來源", safe_text(meta.json_url)],
    ]
    info_table = Table(info_rows, colWidths=[18 * mm, 150 * mm])
    info_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#334155")),
                ("TEXTCOLOR", (1, 0), (1, -1), colors.black),
                ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(info_table)
    story.append(Spacer(1, 6 * mm))

    body = payload.get("JFULL") or ""
    for paragraph in split_body(body):
        story.append(Paragraph(paragraph, body_style))
        story.append(Spacer(1, 1.5 * mm))

    doc.build(story)


def main() -> int:
    parser = argparse.ArgumentParser(description="從裁判資料表搜尋案件並輸出 PDF")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY, help="案件標題關鍵字，例如：請求給付工程款等")
    parser.add_argument("--csv", dest="csv_path", default=CSV_NAME, help="判決總表 CSV 路徑")
    parser.add_argument("--output", dest="output", default=None, help="PDF 輸出檔名或路徑")
    parser.add_argument("--index", type=int, default=1, help="多筆符合時要取第幾筆，從 1 開始")
    parser.add_argument("--font", default=None, help="手動指定中文字型檔案路徑")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"找不到 CSV：{csv_path}", file=sys.stderr)
        return 1

    records = read_records(csv_path)
    matches = find_records(records, args.query)
    if not matches:
        print(f"找不到符合關鍵字的案件：{args.query}", file=sys.stderr)
        return 1

    index = max(1, args.index)
    if index > len(matches):
        print(f"符合案件共有 {len(matches)} 筆，index={index} 超出範圍", file=sys.stderr)
        return 1

    selected = matches[index - 1]
    payload = fetch_json(selected.json_url)

    output_dir = Path(args.output).parent if args.output else Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else output_dir / f"{selected.jid.replace(',', '_')}.pdf"

    font_path = Path(args.font) if args.font else find_font_file()
    font_name = register_font(font_path)

    build_pdf(output_path, selected, payload, font_name)

    print(f"已輸出 PDF：{output_path}")
    print(f"使用案件：{selected.jtitle}")
    print(f"來源 JSON：{selected.json_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
