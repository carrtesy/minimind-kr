# MiniMind-KR

> [!WARNING]
> **🚧 개발 중 (Work In Progress) 🚧**
>
> 이 저장소는 아직 초기 단계입니다. 학습된 모델 가중치, 검증된 파이프라인, 안정화된 API 는 제공되지 않으며, 코드/문서/스크립트 세부 사항이 자주 변경될 수 있습니다. Production 용도로 사용하지 마세요.

> [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 프로젝트를 한국어에 맞게 포팅한 fork입니다.
>
> 원본 저장소의 중국어 자산(시스템 프롬프트, chat template의 tool-calling 안내문, 코드 주석/help 문자열/로그)을 한국어로 지역화했습니다.
> 모델 아키텍처, 학습 파이프라인, 토크나이저 학습 스크립트는 원본을 그대로 계승하며, **한국어 학습 데이터와 토크나이저는 별도로 준비해야 합니다.**

---

## 프로젝트 개요

**MiniMind**는 밑바닥부터 (from scratch) PyTorch만으로 초소형 (~64M dense / ~198M-A64M MoE) 대화 모델의 전체 학습 파이프라인을 재현하는 프로젝트입니다.

- Pretrain → SFT → LoRA → DPO → RLAIF(PPO / GRPO / CISPO) → Agentic RL(Tool Use) → Distillation까지 end-to-end 커버
- 핵심 알고리즘을 `trl`/`peft` 같은 상위 라이브러리 없이 직접 구현
- 아키텍처를 Qwen3 / Qwen3-MoE 생태계와 정렬해 `llama.cpp`/`vllm`/`sglang`/`ollama` 로 export 가능
- 단일 GPU (예: RTX 3090) 로 몇 시간 내 재현 가능한 규모

**MiniMind-KR**은 여기에 한국어 지역화를 얹은 fork입니다.

### 지역화 범위

- 시스템 프롬프트 (`dataset/lm_dataset.py`) — 한국어 전용 10개
- Chat template 및 tool-calling 안내문 (`model/tokenizer_config.json`) — 한국어
- 코드 주석 / docstring / argparse help / 로그 메시지 — 한국어
- Streamlit WebUI (`scripts/web_demo.py`) — 한국어 전용 (다국어 토글 없음)
- 데모 프롬프트 & mock tool 데이터 (`scripts/eval_toolcall.py`, `trainer/train_agent.py`) — 한국 지역 (서울/부산/원화 등)
- `dataset/prepare/` — HuggingFace 공개 한국어 데이터셋을 minimind JSONL 형식으로 변환하는 다운로드 스크립트

### 아직 하지 않은 (당신이 해야 하는) 부분

1. **한국어 토크나이저 재학습 — 필수.**  
   기본 vocab은 6,400 BPE로 한국어 커버리지가 부족합니다. `trainer/train_tokenizer.py` 로 한국어 코퍼스에서 재학습하세요 (README 아래 Quickstart 참조).
2. **모든 가중치 처음부터 학습.**  
   기존 사전학습 가중치는 없습니다. Pretrain → SFT → (선택) DPO / RLAIF / Agent RL 순서로 처음부터 학습해야 합니다.
3. **Reward Model 준비.**  
   PPO/GRPO/Agent RL 스크립트의 `--reward_model_path` 기본값은 플레이스홀더입니다. 한국어 친화 reward model 을 별도로 준비해서 지정해야 합니다.

---

## Quickstart

### 0. 환경

```bash
pip install -r requirements.txt
# PyTorch는 requirements.txt에서 제외되어 있음. 본인 CUDA 버전에 맞게 별도 설치.
```

### 1. 한국어 학습 데이터 준비

이 저장소는 **원본 데이터를 재배포하지 않고, 공개된 HuggingFace 데이터셋을 그때그때 조립해서 사용**합니다.
`dataset/prepare/` 아래에 단계별 다운로드+포맷팅 스크립트가 있습니다.

| 단계 | 스크립트 | 소스 (HuggingFace) | 라이선스 | 출력 파일 |
|---|---|---|---|---|
| Pretrain | `pretrain.py` | `lcw99/wikipedia-korean-20240501` (+ `HuggingFaceFW/fineweb-2`) | Apache-2.0 (+ ODC-BY) | `dataset/pretrain_ko_mini.jsonl` (또는 `pretrain_ko.jsonl` with `--full`) |
| SFT | `sft.py` | `heegyu/open-korean-instructions` (~375k) | MIT | `dataset/sft_ko_mini.jsonl` |
| DPO | `dpo.py` | `maywell/ko_Ultrafeedback_binarized` (~62k) | ⚠️ 데이터 재배포 금지 (모델 배포는 OK) | `dataset/dpo_ko.jsonl` |
| RLAIF | `rlaif.py` | SFT에서 서브샘플 (별도 소스 없음) | — | `dataset/rlaif_ko.jsonl` |
| Agent RL | `agent_rl.py` | `heegyu/glaive-function-calling-v2-ko-mt` (~15k) | Apache-2.0 | `dataset/agent_rl_ko.jsonl` ⚠️ |
| Agent RL Math | `agent_rl_math.py` | `kuotient/orca-math-korean-dpo-pairs` (~193k) | CC-BY-SA-4.0 | `dataset/agent_rl_math_ko.jsonl` |

기본 실행:

```bash
cd dataset/prepare
python pretrain.py                    # Wikipedia ko (~1.5GB) → ../pretrain_ko_mini.jsonl
python sft.py                         # ~375k conversations → ../sft_ko_mini.jsonl
python dpo.py                         # ~62k preference pairs → ../dpo_ko.jsonl
python rlaif.py                       # SFT에서 10k 서브샘플 (sft.py 뒤에)
python agent_rl.py                    # 15k tool-use conversations
python agent_rl_math.py --limit 20000 # math problems + numeric gt
```

각 스크립트는 다음을 지원합니다: `--limit N` (테스트용 서브셋), `--force` (기존 파일 덮어쓰기), `--source <HF path>` (다른 데이터셋으로 교체).

**⚠️ 알려진 한계**
- `agent_rl.py` 는 15k 만 확보 가능 (원본 minimind 는 100k). 또한 `gt` 필드가 빈 문자열로 채워지므로 `train_agent.py` 의 `R_answer` reward 는 항상 0 이고 `R_tool`/`R_format` 만 학습 신호로 작동합니다. 실전에는 `Salesforce/xlam-function-calling-60k` 등을 한국어로 번역하고 LLM 으로 verifier 를 생성해 `gt` 를 채우는 파이프라인이 추가로 필요합니다.
- `HAERAE-HUB/HRM8K` 는 벤치마크로 만들어진 세트라 `agent_rl_math.py` 기본 소스에서 제외했습니다. eval 용으로만 hold-out 하세요.
- `heegyu/open-korean-instructions` 의 실제 스키마가 예상과 다르면 `sft.py` 첫 실행 시 skipped 카운트가 크게 뜹니다. 첫 row 의 keys 가 출력되니 그에 맞춰 `normalize_row` 를 조정하세요.

### 2. 한국어 토크나이저 학습

```bash
cd trainer
python train_tokenizer.py \
    --data_path ../dataset/pretrain_ko_mini.jsonl \
    --tokenizer_dir ../model \
    --vocab_size 8000
```

- `--vocab_size`: 한국어 커버리지를 위해 8,000 ~ 16,000 권장. 원본 6,400 유지도 가능하지만 압축률이 낮아 학습 속도가 떨어집니다.
- `eval_tokenizer` 단계에서 압축률 지표를 확인하세요. 한국어 샘플에서 압축률(문자/토큰)이 너무 낮다면 vocab을 키우세요.

> ⚠️ 토크나이저를 바꾸는 순간 원본에서 제공하는 사전학습 가중치는 재사용 불가입니다. Pretrain부터 다시 돌려야 합니다.

### 3. Pretrain

```bash
cd trainer
python train_pretrain.py --data_path ../dataset/pretrain_ko_mini.jsonl
# 다중 GPU
torchrun --nproc_per_node N train_pretrain.py --data_path ../dataset/pretrain_ko_mini.jsonl
```

산출: `out/pretrain_768.pth`

### 4. SFT

```bash
cd trainer
python train_full_sft.py --data_path ../dataset/sft_ko_mini.jsonl
```

산출: `out/full_sft_768.pth`

### 5. (선택) DPO / LoRA / RLAIF / Agent RL

```bash
python train_dpo.py    --data_path ../dataset/dpo_ko.jsonl
python train_lora.py   --data_path ../dataset/lora_ko_domain.jsonl        # 도메인 데이터는 별도 준비
python train_grpo.py   --data_path ../dataset/rlaif_ko.jsonl              # GRPO / CISPO
python train_ppo.py    --data_path ../dataset/rlaif_ko.jsonl              # PPO
python train_agent.py  --data_path ../dataset/agent_rl_ko.jsonl           # Agentic RL
```

> **Reward Model 참고:** RLAIF/GRPO/PPO/Agent RL 스크립트의 `--reward_model_path` 기본값은 플레이스홀더입니다. 실제 학습 전에 한국어에 적합한 reward model 경로를 지정해야 합니다.

### 6. 추론 / 채팅

```bash
# 저장소 루트에서
python eval_llm.py --weight full_sft
python eval_llm.py --weight full_sft --open_thinking 1     # 적응형 <think> 모드
python eval_llm.py --weight full_sft --lora_weight lora_domain
python eval_llm.py --weight full_sft --inference_rope_scaling   # YaRN 장문 외삽

# OpenAI 호환 서버
cd scripts && python serve_openai_api.py                  # :8998 포트
cd scripts && python chat_api.py                          # 간단 클라이언트

# Streamlit WebUI
cd scripts && streamlit run web_demo.py
```

### 7. 도구 호출 평가

```bash
cd scripts && python eval_toolcall.py --weight full_sft
```

---

## 디렉터리 구조

```
minimind-kr/
├── model/                  # 모델 아키텍처 + tokenizer 파일
│   ├── model_minimind.py   # MiniMindConfig + MiniMindForCausalLM (핵심)
│   ├── model_lora.py       # 자체 구현 LoRA
│   ├── tokenizer.json          ← 한국어로 재학습 필요
│   └── tokenizer_config.json   ← chat_template (한국어 지역화됨)
├── dataset/
│   ├── lm_dataset.py       # PretrainDataset/SFTDataset/DPODataset/RLAIFDataset/AgentRLDataset
│   └── *.jsonl             ← 한국어 데이터 배치
├── trainer/
│   ├── train_tokenizer.py  # 한국어 코퍼스로 tokenizer 재학습 (CLI 지원)
│   ├── train_pretrain.py
│   ├── train_full_sft.py
│   ├── train_dpo.py
│   ├── train_lora.py
│   ├── train_grpo.py       # GRPO / CISPO
│   ├── train_ppo.py
│   ├── train_agent.py      # Agentic RL (multi-turn tool use)
│   ├── train_distillation.py
│   ├── rollout_engine.py   # torch / sglang rollout backend
│   └── trainer_utils.py    # init_model / lm_checkpoint / SkipBatchSampler 등
├── scripts/
│   ├── eval_toolcall.py
│   ├── serve_openai_api.py
│   ├── chat_api.py
│   ├── web_demo.py         # Streamlit UI (한국어 지원)
│   └── convert_model.py    # torch → HF/Qwen3 포맷 변환
├── eval_llm.py             # CLI 채팅 / 평가
└── requirements.txt
```

## 가중치 파일 명명 규칙

`<prefix>_<hidden_size>[_moe].pth`

| Prefix | 스테이지 |
| --- | --- |
| `pretrain` | Pretraining |
| `full_sft` | Full SFT |
| `full_dist` | 지식 증류 |
| `dpo` | DPO |
| `ppo_actor` / `ppo_critic` | PPO |
| `grpo` | GRPO / CISPO |
| `agent` | Agentic RL |
| `lora_<name>` | LoRA 어댑터 |

각 스크립트의 `--save_weight`가 저장할 접두사, `--from_weight`가 로드할 접두사입니다.

## Upstream 동기화

원본 프로젝트가 업데이트되면:

```bash
git fetch upstream
git merge upstream/master
```

지역화된 파일(README, chat_template, lm_dataset.py, argparse help 등)은 conflict가 날 가능성이 높습니다. **한국어 쪽을 유지하면서 upstream의 새 로직만 이식**하는 방향으로 해결하세요.

## 라이선스

원본 프로젝트와 동일하게 [Apache 2.0](./LICENSE).
