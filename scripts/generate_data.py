"""
generate_data.py
────────────────
GPT(gpt-4o-mini)로 token budget 구간별 한국어 공공기관 문서 + 정답 JSON 쌍 생성.

출력: data/raw/raw_{token_budget}_{index}.json
파일 구조:
    {
        "token_budget": 500,
        "doc_type": "보고서",
        "org_type": "지자체",
        "domain": "환경",
        "document": "...",
        "answer": { 기관명, 문서번호, 작성일, 제목, 핵심내용, 담당자, 처리기한 }
    }
"""

import datetime
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import ollama
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_RAW_DIR, DATA_TEST_DIR, EXTRACTION_SCHEMA, GPT_MODEL_GENERATE,
    LOGS_DIR, TEST_LOGS_DIR, OLLAMA_MODELS, TEST_CONFIG,
    TOKEN_BUDGETS, TOKEN_COUNT_TOLERANCE,
)

_TOKEN_COUNT_MODEL = OLLAMA_MODELS[0]  # qwen2.5:1.5b — 토크나이저 기준 모델

load_dotenv(Path(__file__).parent.parent / ".env")

DOC_TYPES = ["공문", "보고서", "규정집", "회의결과", "사업계획서"]
ORG_TYPES = ["중앙부처", "지자체", "공공기관", "교육청"]
DOMAINS   = ["환경", "복지", "교통", "예산", "인사", "시설관리"]

# Qwen 기준 1 토큰 ≈ 2 한국어 글자로 환산한 목표 글자 수
_CHAR_TARGET = {
    200:  400,
    500:  1000,
    1000: 2000,
    2000: 4000,
    4000: 8000,
}

_SCHEMA_STR = json.dumps(EXTRACTION_SCHEMA, ensure_ascii=False, indent=2)

_FEW_SHOT_EXAMPLE = """
## 예시

[입력 문서]
국토교통부 도시정책과
문서번호: 국토도시-2024-0321
작성일: 2024년 3월 21일
담당자: 김민준 사무관
처리기한: 2024년 4월 15일

제목: 2024년 도시재생 활성화 지역 선정 계획 안내

2024년 도시재생 뉴딜 사업의 일환으로 전국 지자체를 대상으로 도시재생 활성화 지역을 공모합니다. 신청 자격은 인구 감소 또는 산업 쇠퇴로 어려움을 겪는 도심지 내 지역으로, 면적은 50만㎡ 이내여야 합니다. 선정된 지역은 최대 5년간 국비 지원을 받으며 주거환경 개선 및 지역경제 활성화 사업을 추진할 수 있습니다.

[정답 JSON]
{
  "기관명": "국토교통부 도시정책과",
  "문서번호": "국토도시-2024-0321",
  "작성일": "2024-03-21",
  "제목": "2024년 도시재생 활성화 지역 선정 계획 안내",
  "핵심내용": "2024년 도시재생 뉴딜 사업 활성화 지역 공모. 인구 감소·산업 쇠퇴 지역 대상, 면적 50만㎡ 이내. 선정 시 최대 5년 국비 지원.",
  "담당자": "김민준 사무관",
  "처리기한": "2024-04-15"
}"""


def _build_prompt(token_budget: int, doc_type: str, org_type: str, domain: str) -> str:
    char_target = _CHAR_TARGET.get(token_budget, token_budget * 2)
    return f"""당신은 한국어 공공기관 문서 생성 전문가입니다.
아래 조건에 따라 한국어 공문서 텍스트와 정답 JSON을 생성하세요.

## 생성 조건
- 문서 유형: {doc_type}
- 기관 유형: {org_type}
- 도메인: {domain}
- 문서 길이: 약 {char_target}자 (공백 포함 한국어 글자 수 기준, plain text, 마크다운·서식 없음)
- 문서 안에 반드시 기관명, 문서번호, 작성일, 제목, 담당자, 처리기한 정보를 포함하세요.
- 핵심내용은 문서 본문에 포함하지 마세요. answer JSON에만 작성합니다.
- 내용은 실제 공문서처럼 구체적이고 자연스럽게 작성하세요.
{_FEW_SHOT_EXAMPLE}

## 추출 대상 JSON Schema
{_SCHEMA_STR}

## 최종 출력 형식 (JSON만, 다른 텍스트 없이)
{{
  "document": "<plain text 문서, 정확히 {token_budget} 토큰>",
  "answer": {{
    "기관명": "...",
    "문서번호": "...",
    "작성일": "YYYY-MM-DD",
    "제목": "...",
    "핵심내용": "...",
    "담당자": "...",
    "처리기한": "YYYY-MM-DD"
  }}
}}"""


def _count_qwen_tokens(text: str) -> int:
    """Qwen 토크나이저 기준 실제 토큰 수 반환 (ollama.generate num_predict=0 활용)."""
    resp = ollama.generate(
        model=_TOKEN_COUNT_MODEL,
        prompt=text,
        options={"num_predict": 0},
    )
    return resp.prompt_eval_count


def _generate_single(client: OpenAI, token_budget: int) -> dict:
    doc_type = random.choice(DOC_TYPES)
    org_type = random.choice(ORG_TYPES)
    domain   = random.choice(DOMAINS)
    prompt   = _build_prompt(token_budget, doc_type, org_type, domain)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL_GENERATE,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.9,
            )
            data = json.loads(response.choices[0].message.content)
            actual_token_count = _count_qwen_tokens(data.get("document", ""))
            data.update(
                token_budget=token_budget,
                doc_type=doc_type,
                org_type=org_type,
                domain=domain,
                actual_token_count=actual_token_count,
            )
            return data
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def _write_generate_log(
    started_at: datetime.datetime,
    samples_per_budget: int,
    seed: int,
    counts: dict,
    test_mode: bool = False,
) -> None:
    finished_at = datetime.datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")

    budgets  = TEST_CONFIG["token_budgets"] if test_mode else TOKEN_BUDGETS
    data_dir = DATA_TEST_DIR if test_mode else DATA_RAW_DIR
    logs_dir = TEST_LOGS_DIR if test_mode else LOGS_DIR
    log_prefix = "test_generate" if test_mode else "generate"

    results_per_budget = {}
    for budget in budgets:
        budget_dir = data_dir / f"budget_{budget}" if not test_mode else data_dir
        token_counts = []
        for i in range(samples_per_budget):
            path = budget_dir / f"raw_{budget}_{i}.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    tc = json.load(f).get("actual_token_count")
                if tc is not None:
                    token_counts.append(tc)

        out_of_range = sum(
            1 for tc in token_counts
            if tc < budget * (1 - TOKEN_COUNT_TOLERANCE) or tc > budget * (1 + TOKEN_COUNT_TOLERANCE)
        )
        token_stats = {}
        if token_counts:
            token_stats = {
                "mean": round(statistics.mean(token_counts), 1),
                "std":  round(statistics.stdev(token_counts) if len(token_counts) > 1 else 0.0, 1),
                "min":  min(token_counts),
                "max":  max(token_counts),
            }

        results_per_budget[str(budget)] = {
            "generated":    counts[budget]["generated"],
            "skipped":      counts[budget]["skipped"],
            "out_of_range": out_of_range,
            "token_stats":  token_stats,
        }

    log = {
        "run_id":      run_id,
        "stage":       "generate_data",
        "test_mode":   test_mode,
        "started_at":  started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "parameters": {
            "gpt_model":             GPT_MODEL_GENERATE,
            "samples_per_budget":    samples_per_budget,
            "seed":                  seed,
            "token_budgets":         budgets,
            "token_count_tolerance": TOKEN_COUNT_TOLERANCE,
        },
        "results_per_budget": results_per_budget,
        "total": {
            "generated":    sum(c["generated"] for c in counts.values()),
            "skipped":      sum(c["skipped"]   for c in counts.values()),
            "out_of_range": sum(v["out_of_range"] for v in results_per_budget.values()),
        },
    }

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{log_prefix}_{run_id}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"Log → {log_path}")


def generate_all(samples_per_budget: int = 20, seed: int = 42, test_mode: bool = False) -> None:
    budgets  = TEST_CONFIG["token_budgets"]      if test_mode else TOKEN_BUDGETS
    n_samples = TEST_CONFIG["samples_per_budget"] if test_mode else samples_per_budget
    data_dir  = DATA_TEST_DIR                     if test_mode else DATA_RAW_DIR

    random.seed(seed)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    started_at = datetime.datetime.now()
    counts = {b: {"generated": 0, "skipped": 0} for b in budgets}

    for budget in budgets:
        print(f"\n[generate] token_budget={budget}tok")
        budget_dir = data_dir / f"budget_{budget}" if not test_mode else data_dir
        budget_dir.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(n_samples), desc=f"{budget}tok"):
            out_path = budget_dir / f"raw_{budget}_{i}.json"
            if out_path.exists():
                counts[budget]["skipped"] += 1
                continue
            try:
                data = _generate_single(client, budget)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                counts[budget]["generated"] += 1
            except Exception as e:
                print(f"  x [{budget}tok] sample {i}: {e}")

    _write_generate_log(started_at, n_samples, seed, counts, test_mode=test_mode)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    generate_all(samples_per_budget=50, seed=100, test_mode=args.test)
