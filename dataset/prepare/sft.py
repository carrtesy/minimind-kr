"""한국어 SFT 데이터 준비.

기본: heegyu/open-korean-instructions (~375k, MIT) - KoAlpaca/ShareGPT-ko/OIG-ko/Korquad-Chat 통합

출력 형식: {"conversations": [{"role": "user"|"assistant"|"system", "content": "..."}]}

실행 (dataset/prepare/ 에서):
  python sft.py                             # 전체
  python sft.py --limit 50000               # 서브셋
  python sft.py --source <다른HF path>       # 다른 SFT 세트로 교체
"""
import argparse
import json
import os
from datasets import load_dataset


def normalize_row(row):
    """다양한 스키마를 minimind conversations 형식으로 정규화."""
    # 이미 conversations 형식
    conv = row.get("conversations")
    if isinstance(conv, list) and conv:
        first = conv[0]
        if isinstance(first, dict) and "role" in first and "content" in first:
            return conv
        # ShareGPT variant: {"from": "human"/"gpt"/"system", "value": "..."}
        if isinstance(first, dict) and "from" in first:
            role_map = {"human": "user", "gpt": "assistant", "system": "system", "tool": "tool"}
            return [{"role": role_map.get(m.get("from"), "user"), "content": m.get("value", "")} for m in conv]

    # messages 형식
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        return [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in msgs]

    # Alpaca 형식 (instruction/input/output)
    instr = row.get("instruction")
    out = row.get("output")
    if instr and out:
        user_content = instr
        if row.get("input"):
            user_content = f"{instr}\n\n{row['input']}"
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": out},
        ]

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="한국어 SFT 데이터 준비")
    parser.add_argument("--output", default="../sft_ko_mini.jsonl")
    parser.add_argument("--source", default="heegyu/open-korean-instructions")
    parser.add_argument("--subset", default=None, help="dataset config name (필요 시)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.force:
        raise SystemExit(f"[skip] {args.output} 이미 존재. --force 로 덮어쓰기 가능")

    print(f"[sft] loading {args.source} ...")
    kwargs = {"split": "train"}
    if args.subset:
        kwargs["name"] = args.subset
    ds = load_dataset(args.source, **kwargs)
    if len(ds) > 0:
        print(f"[sft] first row keys: {list(ds[0].keys())}")

    count, skipped = 0, 0
    with open(args.output, "w", encoding="utf-8") as f:
        for row in ds:
            conv = normalize_row(row)
            if not conv:
                skipped += 1
                continue
            f.write(json.dumps({"conversations": conv}, ensure_ascii=False) + "\n")
            count += 1
            if args.limit and count >= args.limit:
                break
    print(f"[done] {args.output} — {count} rows (skipped {skipped})")
    if skipped and count == 0:
        print("⚠️  모든 rows가 skipped 되었습니다. --source 의 스키마를 확인해 normalize_row 를 수정하세요.")
