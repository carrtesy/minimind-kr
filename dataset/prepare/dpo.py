"""한국어 DPO 데이터 준비.

기본: maywell/ko_Ultrafeedback_binarized (~62k pairs, UltraFeedback 번역)
⚠️ 라이선스: 데이터 재배포 금지 (학습된 모델 배포는 OK). 이 스크립트는 로컬 다운로드만 하고 재업로드하지 않음.

출력 형식: {"chosen": [{"role":..., "content":...}, ...], "rejected": [...]}

실행 (dataset/prepare/ 에서):
  python dpo.py
"""
import argparse
import json
import os
from datasets import load_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="한국어 DPO 데이터 준비")
    parser.add_argument("--output", default="../dpo_ko.jsonl")
    parser.add_argument("--source", default="maywell/ko_Ultrafeedback_binarized")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.force:
        raise SystemExit(f"[skip] {args.output} 이미 존재. --force 로 덮어쓰기 가능")

    print(f"[dpo] loading {args.source} split={args.split} ...")
    ds = load_dataset(args.source, split=args.split)
    if len(ds) > 0:
        print(f"[dpo] first row keys: {list(ds[0].keys())}")

    count, skipped = 0, 0
    with open(args.output, "w", encoding="utf-8") as f:
        for row in ds:
            chosen = row.get("chosen")
            rejected = row.get("rejected")
            # UltraFeedback binarized 는 list of {role, content} 형식
            if not isinstance(chosen, list) or not isinstance(rejected, list):
                skipped += 1
                continue
            f.write(json.dumps({"chosen": chosen, "rejected": rejected}, ensure_ascii=False) + "\n")
            count += 1
            if args.limit and count >= args.limit:
                break
    print(f"[done] {args.output} — {count} pairs (skipped {skipped})")
