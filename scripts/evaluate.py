"""
evaluate.py
───────────
results/result_*.json 파일 450개를 채점하고 두 CSV를 생성한다.

출력:
  results/eval_results.csv        — 샘플별 메트릭 (행 = 결과 파일 1개)
  results/collapse_threshold.csv  — (model, technique)별 안정성 저하 관찰 시점 token_budget
"""

import datetime
import json
import os
import re
import sys
from pathlib import Path

import jsonschema
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ANALYSIS_DIR, ANALYSIS_VERSION, TEST_ANALYSIS_DIR,
    EXPERIMENTS_DIR, TEST_EXPERIMENTS_DIR,
    EXTRACTION_SCHEMA, GPT_MODEL_EVAL,
    LOGS_DIR, TEST_LOGS_DIR,
    TOKEN_BUDGETS,
)

load_dotenv(Path(__file__).parent.parent / ".env")

_REQUIRED = EXTRACTION_SCHEMA["required"]


# ── JSON 추출 ─────────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict | None:
    """raw_output 문자열에서 JSON 객체 추출. 실패 시 None."""
    text = raw.strip()

    # 코드블록 내 JSON 우선 시도
    cb = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if cb:
        text = cb.group(1).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # { 위치를 역방향으로 탐색 (CoT는 JSON이 마지막에 위치)
    for match in reversed(list(re.finditer(r"\{", text))):
        candidate = text[match.start():]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        end = candidate.rfind("}")
        if end != -1:
            try:
                obj = json.loads(candidate[:end + 1])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

    return None


# ── Rule-based 평가 ───────────────────────────────────────────────────────────

def _eval_rule(raw_output: str) -> tuple[bool, bool, float, dict | None]:
    """
    Returns:
        is_valid_json   — JSON 파싱 성공 여부
        schema_valid    — jsonschema 검증 통과 여부
        missing_rate    — 필수 필드 누락 비율 (0.0~1.0)
        parsed          — 파싱된 dict (실패 시 None)
    """
    parsed = _extract_json(raw_output)

    if parsed is None:
        return False, False, 1.0, None

    try:
        jsonschema.validate(instance=parsed, schema=EXTRACTION_SCHEMA)
        schema_valid = True
    except jsonschema.ValidationError:
        schema_valid = False

    missing = [f for f in _REQUIRED if f not in parsed]
    return True, schema_valid, len(missing) / len(_REQUIRED), parsed


# ── 필드별 채점 함수 ──────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """기관명 비교 전 정규화: 괄호 제거, 문장 끝 조사 제거, 공백 정규화."""
    s = re.sub(r'[()（）]', ' ', s)
    s = re.sub(r'(의|에서|으로|로|은|는|이|가)$', '', s.strip())
    return ' '.join(s.split())


def _score_exact(answer_val: str, model_val: str) -> float:
    """문서번호: 공백 정규화 후 exact match."""
    a = answer_val.strip()
    m = model_val.strip()
    if not a:
        return 1.0 if not m else 0.0
    return 1.0 if a == m else 0.0


def _parse_date(s: str):
    """YYYY-MM-DD 또는 YYYY년 MM월 DD일 형식을 date 객체로 변환. 실패 시 None."""
    import datetime
    s = s.strip()
    # YYYY-MM-DD
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    # YYYY년 M월 D일
    m = re.fullmatch(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일", s)
    if m:
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def _score_date(answer_val: str, model_val: str) -> float:
    """작성일·처리기한: 날짜 값이 같으면 1.0, 포맷만 다르면 0.8, 값 틀리거나 파싱 불가면 0.0."""
    a = answer_val.strip()
    m = model_val.strip()
    if not a:
        return 1.0 if not m else 0.0
    if not m:
        return 0.0

    a_date = _parse_date(a)
    if a_date is None:
        return 1.0 if a == m else 0.0  # 정답 파싱 실패 시 exact fallback

    m_date = _parse_date(m)
    if m_date is None:
        return 0.0  # 모델 출력 파싱 실패

    if a_date != m_date:
        return 0.0

    # 날짜 값은 같음 — 포맷 동일 여부 확인
    return 1.0 if a == m else 0.8


def _score_giwanmyeong(answer_val: str, model_val: str) -> float:
    """기관명: 기관+부서명 완전 일치 1.0 / 기관명만 일치(부서명 누락) 0.5 / 그 외 0.0.
    정답에 부서명이 없으면 exact match만으로 1.0 판정.
    비교 전 조사·괄호 정규화(_normalize) 적용.
    """
    a = _normalize(answer_val)
    m = _normalize(model_val)
    if not a:
        return 1.0 if not m else 0.0
    if not m:
        return 0.0
    if a == m:
        return 1.0

    a_parts = a.split()
    # 정답이 단어 1개 = 기관명만 (부서명 없음) → exact만 인정, 이미 위에서 불일치 처리됨
    if len(a_parts) == 1:
        return 0.0

    m_parts = m.split()
    # 모델이 기관명(첫 토큰)만 출력하고 나머지 정답과 기관명이 일치하는 경우 → 0.5
    if len(m_parts) == 1 and m_parts[0] == a_parts[0]:
        return 0.5

    return 0.0


# ── GPT 채점 ──────────────────────────────────────────────────────────────────

def _score_gpt_fields(client: OpenAI, answer: dict, parsed: dict) -> dict[str, float]:
    """담당자·제목·핵심내용 3개 필드를 1회 GPT 호출로 채점. 반환값: 필드명→점수 dict."""
    fields = ["담당자", "제목", "핵심내용"]
    prompt = (
        "아래 세 필드를 각각 채점하여 JSON으로만 출력하라. 설명 없이 JSON만 출력.\n"
        "각 점수는 제시된 선택지 중 하나만 사용하라.\n\n"
        "【담당자 기준】 점수: 1.0 / 0.5 / 0.0\n"
        "- 1.0: 이름 + 직책 모두 일치\n"
        "- 0.5: 이름만 일치, 직책 누락\n"
        "- 0.0: 직책만 있고 이름 없음 / 완전 틀림 / 빈값\n\n"
        "【제목 기준】 점수: 1.0 / 0.5 / 0.0\n"
        "- 1.0: 핵심 단어 모두 포함, 동일\n"
        "- 0.5: 핵심 단어 대부분 있으나 일부 누락·변형\n"
        "- 0.0: 전혀 다름 / 빈값\n\n"
        "【핵심내용 기준】 점수: 1.0 / 0.5 / 0.0\n"
        "순서대로 판단하라:\n"
        "① 정답 핵심내용의 정보 항목을 나열한다\n"
        "② 모델 출력에서 각 항목 포함 여부를 확인한다 (표현이 달라도 내용이 같으면 포함)\n"
        "③ 아래 기준으로 점수를 결정한다:\n"
        "   - 1.0: 핵심 정보 항목이 모두 포함되고 간결하게 요약됨\n"
        "   - 0.5: 핵심 정보 항목 일부만 포함, 또는 원문을 그대로 복사\n"
        "   - 0.0: 핵심 정보 없음 / 완전 오류 / 빈값\n"
        "   (원문 복사 판정: 정답 핵심내용의 핵심 문장이 3개 이상 그대로 반복되면 0.5 상한)\n\n"
        f"정답:\n{json.dumps({f: answer.get(f, '') for f in fields}, ensure_ascii=False, indent=2)}\n\n"
        f"모델 출력:\n{json.dumps({f: parsed.get(f, '') for f in fields}, ensure_ascii=False, indent=2)}\n\n"
        '출력 형식: {"담당자": 점수, "제목": 점수, "핵심내용": 점수}'
    )
    resp = client.chat.completions.create(
        model=GPT_MODEL_EVAL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=60,
    )
    try:
        result = json.loads(resp.choices[0].message.content.strip())
        scores = {}
        for f in fields:
            raw = float(result.get(f, 0.0))
            # 허용 점수 집합으로 클램프 (GPT가 중간값을 출력하는 경우 대비)
            allowed = [0.0, 0.5, 1.0]
            scores[f] = min(allowed, key=lambda x: abs(x - raw))
        return scores
    except Exception:
        return {f: 0.0 for f in fields}


# ── Collapse threshold 계산 ───────────────────────────────────────────────────

_COLLAPSE_THRESHOLDS = [0.10, 0.20, 0.30]


def _first_degradation(grp: pd.DataFrame, metric: str, threshold: float) -> int | str:
    """token_budget 오름차순으로 순회해 실패율이 threshold를 처음 초과하는 budget 반환."""
    for budget in sorted(TOKEN_BUDGETS):
        cell = grp[grp["token_budget"] == budget]
        if cell.empty:
            continue
        if (1 - cell[metric].mean()) > threshold:
            return budget
    return "stable"


def _pattern_type(grp: pd.DataFrame, metric: str) -> str:
    """budget 오름차순 성공률의 첫값·끝값 비교로 패턴 유형 분류.
    recovery: 초반 낮고 후반 높음 (xml 역패턴 등)
    collapse: 초반 높고 후반 낮음 (단조 감소 패턴)
    stable:   차이 없음
    """
    rates = [
        grp[grp["token_budget"] == b][metric].mean()
        for b in sorted(TOKEN_BUDGETS)
        if not grp[grp["token_budget"] == b].empty
    ]
    if len(rates) < 2:
        return "stable"
    if rates[0] < rates[-1]:
        return "recovery"
    if rates[0] > rates[-1]:
        return "collapse"
    return "stable"


def _calc_degradation(df: pd.DataFrame) -> pd.DataFrame:
    """
    (model, technique)별로 is_valid_json / schema_valid 실패율이
    10% / 20% / 30% 임계값을 처음 초과하는 token_budget을 기록.
    4000tok까지 안정적이면 "stable".
    패턴 유형(recovery/collapse/stable)도 함께 기록.
    """
    rows = []
    for (model, technique), grp in df.groupby(["model", "technique"]):
        row = {"model": model, "technique": technique}
        for t in _COLLAPSE_THRESHOLDS:
            pct = int(t * 100)
            row[f"json_degradation_{pct}pct"]   = _first_degradation(grp, "is_valid_json", t)
            row[f"schema_degradation_{pct}pct"] = _first_degradation(grp, "schema_valid",  t)
        row["json_pattern"]   = _pattern_type(grp, "is_valid_json")
        row["schema_pattern"] = _pattern_type(grp, "schema_valid")
        rows.append(row)

    return pd.DataFrame(rows)


# ── Log 작성 ─────────────────────────────────────────────────────────────────

def _write_evaluate_log(
    started_at: datetime.datetime,
    use_gpt: bool,
    files_processed: int,
    df: pd.DataFrame,
    test_mode: bool = False,
) -> None:
    finished_at = datetime.datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")
    log_prefix = "test_evaluate" if test_mode else "evaluate"

    _field_score_cols = [
        "score_기관명", "score_문서번호", "score_작성일",
        "score_제목", "score_핵심내용", "score_담당자", "score_처리기한",
    ]

    def _summarize(sub: pd.DataFrame) -> dict:
        result = {
            "is_valid_json": round(float(sub["is_valid_json"].mean()), 4),
            "schema_valid":  round(float(sub["schema_valid"].mean()), 4),
        }
        if use_gpt and sub["semantic_score"].notna().any():
            result["semantic_score"] = round(float(sub["semantic_score"].mean()), 4)
            result["field_scores"] = {
                col.replace("score_", ""): round(float(sub[col].mean()), 4)
                for col in _field_score_cols
                if col in sub.columns and sub[col].notna().any()
            }
        return result

    log = {
        "run_id":      run_id,
        "stage":       "evaluate",
        "test_mode":   test_mode,
        "started_at":  started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "parameters": {
            "use_gpt":   use_gpt,
            "gpt_model": GPT_MODEL_EVAL if use_gpt else None,
        },
        "files_processed": files_processed,
        "output_files": [
            "eval_results.csv",
            "collapse_threshold.csv",
        ],
        "summary_by_model": {
            model: _summarize(df[df["model"] == model])
            for model in sorted(df["model"].unique())
        },
        "summary_by_technique": {
            tech: _summarize(df[df["technique"] == tech])
            for tech in sorted(df["technique"].unique())
        },
    }

    logs_dir = TEST_LOGS_DIR if test_mode else LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{log_prefix}_{run_id}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"Log → {log_path}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def evaluate_all(use_gpt: bool = True, test_mode: bool = False) -> pd.DataFrame:
    exp_dir     = TEST_EXPERIMENTS_DIR if test_mode else EXPERIMENTS_DIR
    analysis_dir = TEST_ANALYSIS_DIR   if test_mode else ANALYSIS_DIR / ANALYSIS_VERSION
    analysis_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if use_gpt else None
    rows = []

    started_at = datetime.datetime.now()

    test_subdir = exp_dir / "test"
    result_files = sorted(
        f for f in exp_dir.rglob("result_*.json")
        if not f.is_relative_to(test_subdir)
    )
    if not result_files:
        print("결과 파일 없음. run_experiment.py를 먼저 실행하세요.")
        return pd.DataFrame()

    for path in tqdm(result_files, desc="evaluating"):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)

        is_valid, schema_ok, missing_rate, parsed = _eval_rule(r.get("raw_output", ""))
        answer = r["answer"]

        if not use_gpt or client is None:
            field_scores = None
        elif not is_valid:
            field_scores = {f: 0.0 for f in ["기관명", "문서번호", "작성일", "제목", "핵심내용", "담당자", "처리기한"]}
        else:
            # 필드 값이 string이 아닌 경우(list 등) 빈 문자열로 정규화
            p = {
                k: v if isinstance(v, str) else ""
                for k, v in (parsed or {}).items()
            }
            code_scores = {
                "문서번호": _score_exact(answer.get("문서번호", ""), p.get("문서번호", "")),
                "작성일":   _score_date(answer.get("작성일", ""),   p.get("작성일", "")),
                "처리기한": _score_date(answer.get("처리기한", ""), p.get("처리기한", "")),
                "기관명":   _score_giwanmyeong(answer.get("기관명", ""), p.get("기관명", "")),
            }
            gpt_scores = _score_gpt_fields(client, answer, p)
            field_scores = {**code_scores, **gpt_scores}

        semantic = (
            None if field_scores is None
            else round(sum(field_scores.values()) / len(field_scores), 4)
        )

        rows.append({
            "model":               r["model"],
            "technique":           r["technique"],
            "token_budget":        r["token_budget"],
            "index":               r["index"],
            "is_valid_json":       is_valid,
            "schema_valid":        schema_ok,
            "missing_field_rate":  missing_rate,
            "score_기관명":        field_scores.get("기관명")   if field_scores else None,
            "score_문서번호":      field_scores.get("문서번호") if field_scores else None,
            "score_작성일":        field_scores.get("작성일")   if field_scores else None,
            "score_제목":          field_scores.get("제목")     if field_scores else None,
            "score_핵심내용":      field_scores.get("핵심내용") if field_scores else None,
            "score_담당자":        field_scores.get("담당자")   if field_scores else None,
            "score_처리기한":      field_scores.get("처리기한") if field_scores else None,
            "semantic_score":      semantic,
            "latency_sec":         r.get("latency_sec"),
            "prompt_token_count":  r.get("prompt_token_count"),
            "output_token_count":  r.get("output_token_count"),
        })

    df = pd.DataFrame(rows)

    eval_path = analysis_dir / "eval_results.csv"
    df.to_csv(eval_path, index=False, encoding="utf-8-sig")
    print(f"Saved → {eval_path}")

    collapse_df = _calc_degradation(df)
    collapse_path = analysis_dir / "collapse_threshold.csv"
    collapse_df.to_csv(collapse_path, index=False, encoding="utf-8-sig")
    print(f"Saved → {collapse_path}")

    _write_evaluate_log(started_at, use_gpt, len(result_files), df, test_mode=test_mode)

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--no-gpt", action="store_true")
    args = parser.parse_args()
    evaluate_all(use_gpt=not args.no_gpt, test_mode=args.test)
