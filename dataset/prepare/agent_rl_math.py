"""Agent RL Math (RLVR) 데이터 준비 - 검증가능한 숫자 답.

기본: kuotient/orca-math-korean-dpo-pairs (~193k pairs, CC-BY-SA-4.0)
chosen (정답 풀이) 마지막에서 숫자를 추출해 gt 로 저장.

⚠️  HAERAE-HUB/HRM8K 는 학습에 사용 금지 (벤치마크). eval 용으로만 hold out.

출력 형식: {"conversations": [{"role":"user","content":문제}], "gt": "정답숫자"}

실행 (dataset/prepare/ 에서):
  python agent_rl_math.py --limit 20000
"""
import argparse
import json
import os
import re
from datasets import load_dataset

# chosen 응답 끝에서 숫자 추출 (음수/소수/천단위 콤마 대응)
_ANS_RE = re.compile(r"([-]?\d[\d,]*(?:\.\d+)?)(?![\d.])")


def extract_answer(text):
    if not text:
        return ""
    matches = _ANS_RE.findall(text)
    if not matches:
        return ""
    return matches[-1].replace(",", "")


def extract_chosen_text(chosen):
    """chosen 이 str 이면 그대로, list 이면 마지막 assistant 메시지."""
    if isinstance(chosen, str):
        return chosen
    if isinstance(chosen, list):
        for m in reversed(chosen):
            if isinstance(m, dict) and m.get("role") == "assistant":
                return m.get("content", "")
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent RL Math (RLVR) 데이터 준비")
    parser.add_argument("--output", default="../agent_rl_math_ko.jsonl")
    parser.add_argument("--source", default="kuotient/orca-math-korean-dpo-pairs")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.force:
        raise SystemExit(f"[skip] {args.output} 이미 존재. --force 로 덮어쓰기 가능")

    print(f"[math] loading {args.source} ...")
    ds = load_dataset(args.source, split="train")
    if len(ds) > 0:
        print(f"[math] first row keys: {list(ds[0].keys())}")

    count, skipped = 0, 0
    with open(args.output, "w", encoding="utf-8") as f:
        for row in ds:
            question = row.get("question") or row.get("prompt") or row.get("instruction") or ""
            chosen_txt = extract_chosen_text(row.get("chosen") or row.get("output") or row.get("answer"))
            if not question or not chosen_txt:
                skipped += 1
                continue
            gt = extract_answer(chosen_txt)
            if not gt:
                skipped += 1
                continue
            record = {
                "conversations": [{"role": "user", "content": question}],
                "gt": gt,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if args.limit and count >= args.limit:
                break
    print(f"[done] {args.output} — {count} rows (skipped {skipped})")
