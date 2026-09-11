# minimind-kr 데이터셋

`dataset/prepare/` 아래의 스크립트로 공개 HuggingFace 한국어 데이터셋을
minimind 학습 형식(JSONL) 으로 변환해 이 디렉터리에 저장합니다.
이 저장소는 원본 데이터를 재배포하지 않습니다.

준비 절차:

```bash
cd dataset/prepare
python pretrain.py       # → ../pretrain_ko_mini.jsonl
python sft.py            # → ../sft_ko_mini.jsonl
python dpo.py            # → ../dpo_ko.jsonl
python rlaif.py          # → ../rlaif_ko.jsonl  (sft.py 뒤에)
python agent_rl.py       # → ../agent_rl_ko.jsonl
python agent_rl_math.py  # → ../agent_rl_math_ko.jsonl
```

소스 데이터셋, 라이선스 주의사항, 알려진 한계는 `README.md` 의
"1. 한국어 학습 데이터 준비" 섹션과 각 스크립트 상단 docstring 을 참고.

생성된 `*.jsonl` 파일은 `.gitignore` 로 제외되어 커밋되지 않습니다.
