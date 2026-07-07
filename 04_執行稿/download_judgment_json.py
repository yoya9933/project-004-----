from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


BASE_URL = "http://140.116.245.86:8090"
DEFAULT_QUERY = "請求給付工程款等"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "02_輸入資料" / "json_output"
USER_AGENT = "Mozilla/5.0 (compatible; judgment-json-downloader/1.0)"


@dataclass(frozen=True)
class JudgmentMeta:
    month: str
    court: str
    jid: str
    title: str
    case_code: str
    case_no: str
    date: str
    char_len: int
    via: str
    pdf: str
    json_url: str

    @property
    def relative_json_path(self) -> str:
        return f"{self.month}/{self.court}/{self.jid}.json"


def fetch_bytes(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_json(url: str, timeout: int) -> Any:
    data = fetch_bytes(url, timeout)
    return json.loads(data.decode("utf-8"))


def join_url(base_url: str, relative_path: str) -> str:
    base = base_url.rstrip("/") + "/"
    encoded_path = urllib.parse.quote(relative_path.replace("\\", "/"), safe="/,")
    return urllib.parse.urljoin(base, encoded_path)


def flatten_manifest(manifest: dict[str, Any], base_url: str) -> list[JudgmentMeta]:
    records: list[JudgmentMeta] = []
    months = manifest.get("months") or {}
    for month, courts in months.items():
        if not isinstance(courts, dict):
            continue
        for court, items in courts.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                jid = str(item.get("id") or "").strip()
                if not jid:
                    continue
                relative_path = f"{month}/{court}/{jid}.json"
                records.append(
                    JudgmentMeta(
                        month=str(month),
                        court=str(court),
                        jid=jid,
                        title=str(item.get("t") or ""),
                        case_code=str(item.get("case") or ""),
                        case_no=str(item.get("no") or ""),
                        date=str(item.get("date") or ""),
                        char_len=int(item.get("len") or 0),
                        via=str(item.get("via") or ""),
                        pdf=str(item.get("pdf") or ""),
                        json_url=join_url(base_url, relative_path),
                    )
                )
    return records


def searchable_text(record: JudgmentMeta, fields: str) -> str:
    if fields == "title":
        parts = [record.title]
    elif fields == "all":
        parts = [
            record.title,
            record.case_code,
            record.case_no,
            record.jid,
            record.court,
            record.month,
            record.date,
            record.via,
        ]
    else:
        # Same fields used by the web UI search box.
        parts = [record.title, record.case_code, record.case_no, record.jid]
    return " ".join(parts).casefold()


def find_matches(
    records: Iterable[JudgmentMeta],
    query: str,
    fields: str,
    exact_title: bool,
) -> list[JudgmentMeta]:
    query = query.strip()
    if not query:
        return []
    if exact_title:
        return [record for record in records if record.title == query]
    needle = query.casefold()
    return [record for record in records if needle in searchable_text(record, fields)]


def safe_name(value: str, fallback: str = "output", max_len: int = 80) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value.strip())
    cleaned = re.sub(r"\s+", "_", cleaned).strip(" ._")
    if not cleaned:
        cleaned = fallback
    return cleaned[:max_len]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def local_raw_path(output_dir: Path, record: JudgmentMeta) -> Path:
    return output_dir / "raw" / record.month / record.court / f"{record.jid}.json"


def download_matches(
    matches: list[JudgmentMeta],
    output_dir: Path,
    timeout: int,
    sleep_seconds: float,
    strict: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    downloaded: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for index, record in enumerate(matches, start=1):
        print(f"[{index}/{len(matches)}] 下載 {record.date} {record.court} {record.title} {record.jid}")
        try:
            payload = fetch_json(record.json_url, timeout=timeout)
            path = local_raw_path(output_dir, record)
            write_json(path, payload)
            downloaded.append(
                {
                    "metadata": asdict(record),
                    "local_json": str(path),
                    "payload": payload,
                }
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            error = {
                "jid": record.jid,
                "title": record.title,
                "json_url": record.json_url,
                "error": str(exc),
            }
            errors.append(error)
            print(f"  失敗：{record.json_url} ({exc})", file=sys.stderr)
            if strict:
                break
        if sleep_seconds > 0 and index < len(matches):
            time.sleep(sleep_seconds)

    return downloaded, errors


def build_index(
    query: str,
    base_url: str,
    fields: str,
    exact_title: bool,
    matches: list[JudgmentMeta],
    downloaded: list[dict[str, Any]],
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    local_by_jid = {
        item["metadata"]["jid"]: item["local_json"]
        for item in downloaded
    }
    return {
        "base_url": base_url,
        "query": query,
        "fields": fields,
        "exact_title": exact_title,
        "matched_count": len(matches),
        "downloaded_count": len(downloaded),
        "error_count": len(errors),
        "items": [
            {
                **asdict(record),
                "local_json": local_by_jid.get(record.jid, ""),
            }
            for record in matches
        ],
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="搜尋工程裁判資料庫 manifest，並下載符合條件的裁判原文 JSON。"
    )
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY, help="搜尋關鍵字")
    parser.add_argument("--base-url", default=BASE_URL, help="資料庫網址")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="輸出資料夾")
    parser.add_argument(
        "--fields",
        choices=("ui", "title", "all"),
        default="ui",
        help="搜尋欄位：ui 會模擬網頁搜尋框；title 只搜案由；all 搜全部中繼資料",
    )
    parser.add_argument("--exact-title", action="store_true", help="案由必須完全等於搜尋字串")
    parser.add_argument("--limit", type=int, default=0, help="只下載前 N 筆；0 表示全部")
    parser.add_argument("--timeout", type=int, default=30, help="單次 HTTP 逾時秒數")
    parser.add_argument("--sleep", type=float, default=0.05, help="每筆下載間隔秒數")
    parser.add_argument("--strict", action="store_true", help="遇到第一個下載錯誤就停止")
    parser.add_argument("--no-combined", action="store_true", help="不輸出含全文 payload 的 combined.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    output_dir = Path(args.output_dir) / safe_name(args.query, fallback="query")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_url = join_url(base_url, "manifest.json")
    print(f"讀取 manifest：{manifest_url}")
    manifest = fetch_json(manifest_url, timeout=args.timeout)
    records = flatten_manifest(manifest, base_url=base_url)
    print(f"manifest 共 {len(records)} 筆中繼資料")

    matches = find_matches(
        records,
        query=args.query,
        fields=args.fields,
        exact_title=args.exact_title,
    )
    matches.sort(key=lambda item: (item.date, item.court, item.jid))
    if args.limit and args.limit > 0:
        matches = matches[: args.limit]

    if not matches:
        print(f"找不到符合「{args.query}」的裁判。", file=sys.stderr)
        return 1

    print(f"符合「{args.query}」共 {len(matches)} 筆，輸出到：{output_dir}")
    downloaded, errors = download_matches(
        matches,
        output_dir=output_dir,
        timeout=args.timeout,
        sleep_seconds=args.sleep,
        strict=args.strict,
    )

    index = build_index(
        query=args.query,
        base_url=base_url,
        fields=args.fields,
        exact_title=args.exact_title,
        matches=matches,
        downloaded=downloaded,
        errors=errors,
    )
    index_path = output_dir / "index.json"
    write_json(index_path, index)
    print(f"已輸出索引：{index_path}")

    if not args.no_combined:
        combined_path = output_dir / "combined.json"
        write_json(
            combined_path,
            {
                **{key: value for key, value in index.items() if key != "items"},
                "items": downloaded,
            },
        )
        print(f"已輸出合併檔：{combined_path}")

    if errors:
        print(f"完成，但有 {len(errors)} 筆失敗，詳見 index.json 的 errors。", file=sys.stderr)
        return 2
    print(f"完成，成功下載 {len(downloaded)} 筆 JSON。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
