"""Agentic RL 학습 데이터 준비 (multi-turn tool-use).

⚠️  스타터 스크립트. 다음 두 가지 한계가 있음:
    1. 규모: heegyu/glaive-function-calling-v2-ko-mt 는 ~15k rows (원본 agent_rl.jsonl ~100k 대비).
       원본 규모가 필요하면 Salesforce/xlam-function-calling-60k 등을 한국어로 번역 필요.
    2. gt 필드: 이 데이터셋에는 reward 검증용 ground-truth 가 없음. 기본으로 빈 문자열 저장.
       train_agent.py 의 R_answer 항목은 이 상태에서 항상 0 이며, R_tool/R_format 만 학습 신호로 작동.
       실전에는 LLM 으로 각 trajectory 의 verifier (regex/JSON 검증) 를 생성해 gt 를 채워야 함.

출력 형식: {"conversations": [...], "gt": ""}

실행 (dataset/prepare/ 에서):
  python agent_rl.py
"""
import argparse
import json
import os
from datasets import load_dataset


def parse_row(row):
    """Glaive-ko-mt row → (conversations, tools). 스키마가 다양해 방어적으로 처리."""
    # 이미 conversations/messages 형식이면 그대로 사용
    for key in ("conversations", "messages"):
        val = row.get(key)
        if isinstance(val, list) and val:
            first = val[0]
            if isinstance(first, dict) and "role" in first:
                return val, row.get("tools") or row.get("functions")
            # ShareGPT-style
            if isinstance(first, dict) and "from" in first:
                role_map = {"human": "user", "gpt": "assistant", "system": "system", "tool": "tool", "function_call": "assistant", "observation": "tool"}
                return [{"role": role_map.get(m.get("from"), "user"), "content": m.get("value", "")} for m in val], row.get("tools")

    return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic RL 데이터 준비 (스타터)")
    parser.add_argument("--output", default="../agent_rl_ko.jsonl")
    parser.add_argument("--source", default="heegyu/glaive-function-calling-v2-ko-mt")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.force:
        raise SystemExit(f"[skip] {args.output} 이미 존재. --force 로 덮어쓰기 가능")

    print(f"[agent_rl] loading {args.source} ...")
    ds = load_dataset(args.source, split="train")
    if len(ds) > 0:
        print(f"[agent_rl] first row keys: {list(ds[0].keys())}")
        sample_str = json.dumps(dict(ds[0]), ensure_ascii=False)
        print(f"[agent_rl] sample (첫 300자): {sample_str[:300]}")

    count, skipped = 0, 0
    with open(args.output, "w", encoding="utf-8") as f:
        for row in ds:
            conv, tools = parse_row(row)
            if not conv:
                skipped += 1
                continue
            # tools 는 minimind 규약대로 system 메시지에 붙임
            if tools:
                tools_str = tools if isinstance(tools, str) else json.dumps(tools, ensure_ascii=False)
                sys_msg = next((m for m in conv if m.get("role") == "system"), None)
                if sys_msg is not None:
                    sys_msg["tools"] = tools_str
                else:
                    conv = [{"role": "system", "content": "", "tools": tools_str}] + conv
            f.write(json.dumps({"conversations": conv, "gt": ""}, ensure_ascii=False) + "\n")
            count += 1
            if args.limit and count >= args.limit:
                break
    print(f"[done] {args.output} — {count} rows (skipped {skipped})")
    if count > 0:
        print("⚠️  gt 필드가 빈 문자열입니다. train_agent.py 의 answer reward 는 이 상태로 0.")
        print("    자세한 내용은 README.md 의 '데이터 준비' 섹션 참고.")
