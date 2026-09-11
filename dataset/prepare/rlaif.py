"""RLAIF 학습용 프롬프트 풀.

전용 한국어 RLAIF 세트가 없으므로 SFT 결과에서 서브샘플링한다.
(원본 minimind 도 rlaif.jsonl 을 SFT 서브샘플로 만들었음.)

출력 형식: SFT와 동일한 {"conversations": [...]} JSONL

실행 (dataset/prepare/ 에서, sft.py 먼저 실행 필요):
  python rlaif.py                                          # 기본: 10000
  python rlaif.py --input ../sft_ko.jsonl --n 20000
"""
import argparse
import os
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLAIF 프롬프트 풀 준비 (SFT 서브샘플)")
    parser.add_argument("--input", default="../sft_ko_mini.jsonl", help="입력 SFT JSONL")
    parser.add_argument("--output", default="../rlaif_ko.jsonl")
    parser.add_argument("--n", type=int, default=10000, help="샘플링할 rows 수")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"[error] {args.input} 없음. 먼저 sft.py 실행하세요.")
    if os.path.exists(args.output) and not args.force:
        raise SystemExit(f"[skip] {args.output} 이미 존재. --force 로 덮어쓰기 가능")

    with open(args.input, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    print(f"[rlaif] loaded {len(lines)} rows from {args.input}")

    random.seed(args.seed)
    sampled = random.sample(lines, min(args.n, len(lines)))

    with open(args.output, "w", encoding="utf-8") as f:
        for ln in sampled:
            f.write(ln)
    print(f"[done] {args.output} — {len(sampled)} rows")
