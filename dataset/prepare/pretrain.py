"""한국어 pretrain 데이터 준비.

기본: lcw99/wikipedia-korean-20240501 (~1.75GB, Apache-2.0)
--full: + HuggingFaceFW/fineweb-2 kor_Hang 스트리밍 (ODC-BY)

출력 형식: {"text": "..."} JSONL

실행 (dataset/prepare/ 에서):
  python pretrain.py                        # mini (~1.5GB) → ../pretrain_ko_mini.jsonl
  python pretrain.py --full                 # + fineweb 1M rows → ../pretrain_ko.jsonl
  python pretrain.py --limit 10000          # 테스트용 짧게
"""
import argparse
import json
import os
from datasets import load_dataset


def dump_wikipedia(writer, limit=0):
    print("[wiki] loading lcw99/wikipedia-korean-20240501 ...")
    ds = load_dataset("lcw99/wikipedia-korean-20240501", split="train")
    count = 0
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        writer.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        count += 1
        if limit and count >= limit:
            break
    print(f"[wiki] wrote {count} rows")
    return count


def dump_fineweb(writer, limit):
    print(f"[fineweb-2] streaming kor_Hang (limit={limit}) ...")
    ds = load_dataset("HuggingFaceFW/fineweb-2", name="kor_Hang", split="train", streaming=True)
    count = 0
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        writer.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        count += 1
        if limit and count >= limit:
            break
    print(f"[fineweb-2] wrote {count} rows")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="한국어 pretrain 데이터 준비")
    parser.add_argument("--output", default=None, help="출력 JSONL 경로 (기본: mini/full에 따라 자동)")
    parser.add_argument("--full", action="store_true", help="Wikipedia + FineWeb-2 스트리밍 (기본은 Wikipedia만)")
    parser.add_argument("--limit", type=int, default=0, help="Wikipedia 최대 rows (0=제한 없음)")
    parser.add_argument("--fineweb_limit", type=int, default=1_000_000, help="--full 시 FineWeb 최대 rows")
    parser.add_argument("--force", action="store_true", help="기존 파일 덮어쓰기")
    args = parser.parse_args()

    out = args.output or ("../pretrain_ko.jsonl" if args.full else "../pretrain_ko_mini.jsonl")
    if os.path.exists(out) and not args.force:
        raise SystemExit(f"[skip] {out} 이미 존재. --force 로 덮어쓰기 가능")

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        total = dump_wikipedia(f, limit=args.limit)
        if args.full:
            total += dump_fineweb(f, limit=args.fineweb_limit)
    print(f"[done] {out} — {total} rows")
