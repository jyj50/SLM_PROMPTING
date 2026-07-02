# SLM Prompting Stability Research

소형 언어모델(qwen2.5 1.5B / 3B / 7B)에서 **prompting technique × 문서 길이 조합별 JSON 구조화 출력 안정성**을 정량적으로 측정하고, 각 token_budget 조건에서 technique별 안정성 차이를 수치로 도출하는 실험 연구입니다.

---

## 핵심 질문

1. Prompting technique가 달라질 때 소형 모델의 JSON 구조화 출력이 얼마나 안정적으로 나오는가?
2. 각 technique는 token_budget 조건별로 안정성이 어떻게 다른가?

---

## 실험 설계

| 구분 | 항목 |
|---|---|
| **모델** | qwen2.5:1.5b / qwen2.5:3b / qwen2.5:7b |
| **Prompting technique** | zero_shot, few_shot, cot, structured, xml, json_schema |
| **문서 길이 (토큰)** | 200 / 500 / 1000 / 2000 / 4000 |
| **셀당 샘플 수** | 50개 문서 |
| **총 모델 호출** | 3 × 6 × 5 × 50 = 4,500회 |

**측정 메트릭:**
- `is_valid_json` — JSON 파싱 성공 여부
- `schema_valid` — 7개 필드 구조 일치 여부
- `missing_field_rate` — 필드 키 누락 비율 (0.0~1.0)
- `semantic_score` — 필드별 채점 평균 (0.0~1.0)
- `score_{필드명}` — 필드별 개별 점수 (기관명·문서번호·작성일·제목·핵심내용·담당자·처리기한)
- `latency_sec` — 추론 시간 (초)
- `prompt_token_count` — 입력 토큰 수
- `output_token_count` — 출력 토큰 수

---

## 프로젝트 구조

```
SLM_prompting/
├── config.py                         # 실험 전반 상수 (모델, 경로, schema 등)
├── main.py
├── README.md
├── CLAUDE.md
├── .env                              # OPENAI_API_KEY (git 제외)
│
├── data/
│   ├── raw/                          # 실험 입력 문서
│   │   ├── budget_200/               # 토큰 구간별 서브폴더 (각 50개)
│   │   ├── budget_500/
│   │   ├── budget_1000/
│   │   ├── budget_2000/
│   │   └── budget_4000/
│   └── test/                         # 테스트용 소량 문서
│
├── experiments/                      # 실험 원본 출력 (모델 호출 결과 JSON)
│   ├── qwen2.5-1.5b/
│   │   ├── cot/                      # 기법별 서브폴더 (각 250개)
│   │   ├── few_shot/
│   │   ├── structured/
│   │   ├── xml/
│   │   ├── json_schema/
│   │   └── zero_shot/
│   ├── qwen2.5-3b/
│   ├── qwen2.5-7b/
│   └── test/                         # 테스트 실험 결과
│
├── analysis/                         # 채점 및 분석 결과
│   ├── v1/                           # 1차 채점 (GPT 단일 호출 방식)
│   │   ├── SCORING.md                # v1 채점 기준·결과·한계 설명
│   │   ├── eval_results.csv
│   │   ├── collapse_threshold.csv
│   │   └── plot_*.png
│   ├── v2/                           # 2차 채점 (필드별 채점 방식, 현재)
│   │   ├── SCORING.md                # v2 채점 기준·결과 설명
│   │   ├── eval_results.csv
│   │   ├── collapse_threshold.csv
│   │   └── plot_*.png
│   └── test/                         # 테스트 분석 결과
│
├── scripts/
│   ├── generate_data.py              # 1단계: GPT로 데이터 생성
│   ├── run_experiment.py             # 2단계: 소형 모델 실험 실행
│   ├── evaluate.py                   # 3단계: rule-based + GPT 평가
│   └── report.py                     # 4단계: 시각화
│
├── prompts/                          # 기법별 프롬프트 템플릿
│   ├── __init__.py
│   ├── zero_shot.txt
│   ├── few_shot.txt
│   ├── cot.txt
│   ├── structured.txt
│   ├── xml.txt
│   └── json_schema.txt
│
└── logs/
    ├── production/                   # 본 실험 실행 로그
    └── test/                         # 테스트 실행 로그
```

---

## 추출 대상 JSON Schema

한국어 공공기관 문서에서 아래 7개 필드를 추출합니다.

```json
{
  "기관명":   "발신 기관 또는 담당 부서명",
  "문서번호": "문서 고유 번호",
  "작성일":   "작성일 (YYYY-MM-DD)",
  "제목":     "문서 제목",
  "핵심내용": "문서 핵심 내용 요약 (1~3문장)",
  "담당자":   "담당자 이름 (직책 포함)",
  "처리기한": "처리 또는 제출 기한 (YYYY-MM-DD)"
}
```

---

## 실행 순서

### 사전 준비

`.env` 파일에 API 키 설정:
```
OPENAI_API_KEY=sk-...
```

Ollama 모델 설치:
```bash
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
```

### 테스트 실행 (파이프라인 검증)

모델 1개 × technique 2개 × budget 2구간 × 샘플 2개 = **8회 호출**로 전체 흐름 확인.

```bash
uv run python scripts/generate_data.py --test
uv run python scripts/run_experiment.py --test
uv run python scripts/evaluate.py --test --no-gpt
uv run python scripts/report.py --test
```

테스트 결과는 `data/test/`, `experiments/test/`, `analysis/test/`에 저장됩니다.

### 실제 실험 실행

```bash
uv run python scripts/generate_data.py
uv run python scripts/run_experiment.py
uv run python scripts/evaluate.py
uv run python scripts/report.py
```

각 단계는 독립적으로 재실행 가능합니다. 이미 존재하는 파일은 건너뜁니다.

---

## 단계별 설명

### 1단계: 데이터 생성 (`generate_data.py`)

GPT로 token budget별 한국어 공공기관 문서 + 정답 JSON 쌍을 생성합니다.

- 출력: `data/raw/budget_{N}/raw_{budget}_{index}.json`
- 완료 시 `logs/production/generate_{timestamp}.json` 자동 생성

### 2단계: 실험 실행 (`run_experiment.py`)

4,500개 셀(모델 × technique × budget × 샘플)을 순회하며 Ollama 로컬 모델을 호출합니다.

- 출력: `experiments/{model}/{technique}/result_{model}_{technique}_{budget}_{index}.json`
- 완료 시 `logs/production/experiment_{timestamp}.json` 자동 생성

### 3단계: 평가 (`evaluate.py`)

필드별 채점을 실행합니다. 날짜·문서번호·기관명은 코드로, 담당자·제목·핵심내용은 GPT로 채점합니다.

- 출력: `analysis/v2/eval_results.csv` — 샘플별 전체 메트릭 (4500행)
- 출력: `analysis/v2/collapse_threshold.csv` — (model, technique)별 안정성 저하 관찰 시점 token_budget (10%/20%/30% 기준)
- `--no-gpt` 사용 시 `semantic_score`는 `None`으로 기록
- 완료 시 `logs/production/evaluate_{timestamp}.json` 자동 생성

### 4단계: 시각화 (`report.py`)

```bash
uv run python scripts/report.py
```

- 출력 (`analysis/v2/` 아래):
  - `plot_valid_json_rate.png` — JSON 유효율
  - `plot_schema_valid_rate.png` — Schema 준수율
  - `plot_semantic_score.png` — 의미적 정확도
  - `plot_latency.png` — token budget별 추론 시간
  - `plot_field_bar.png` — 기법별 필드 점수 bar chart
  - `plot_field_line.png` — 필드별 token_budget × 점수 라인 플롯

---

## 채점 버전 이력

| 버전 | 채점 방식 | 위치 |
|---|---|---|
| v1 | GPT 1회 호출로 7개 필드 단일 점수 | `analysis/v1/` |
| v2 | 필드별 개별 채점 (코드 4개 + GPT 3개) | `analysis/v2/` |

버전별 상세 내용은 각 폴더의 `SCORING.md` 참조.

---

## 환경

- Python 3.10+ / 패키지 관리: `uv`
- 의존성: `openai`, `ollama`, `python-dotenv`, `pandas`, `jsonschema`, `tqdm`, `matplotlib`
- 로컬 추론: Ollama (qwen2.5:1.5b · 3b · 7b)
- 데이터 생성: OpenAI API (`gpt-5.4-mini-2026-03-17`)
- 의미 채점: OpenAI API (`gpt-4o-mini`)
