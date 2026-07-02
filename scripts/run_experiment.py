"""
run_experiment.py
─────────────────
90개 셀(모델 × technique × token_budget)을 순회하며 Ollama 로컬 모델 실험 실행.

입력:  data/raw/raw_{token_budget}_{index}.json
출력:  results/result_{model}_{technique}_{token_budget}_{index}.json
저장:  model, technique, token_budget, index, input_document, answer, raw_output
"""

import datetime
import json
import sys
import time
from pathlib import Path

import ollama
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_RAW_DIR, DATA_TEST_DIR,
    EXPERIMENTS_DIR, TEST_EXPERIMENTS_DIR,
    EXTRACTION_SCHEMA,
    LOGS_DIR, TEST_LOGS_DIR,
    OLLAMA_MODELS,
    PROMPTING_TECHNIQUES,
    RUNS_PER_CELL,
    TEST_CONFIG,
    TOKEN_BUDGETS,
)
from prompts import load_prompt_template


def _model_slug(model: str) -> str:
    """파일명용 모델 식별자.  qwen2.5:1.5b → qwen2.5-1.5b"""
    return model.replace(":", "-")


def _call_model(model: str, prompt: str) -> tuple[str, dict]:
    """
    Ollama 모델 호출. 실패 시 최대 3회 재시도 (지수 백오프).
    Returns: (raw_output, meta)
      meta keys: latency_sec, prompt_token_count, output_token_count
    """
    for attempt in range(3):
        try:
            t0 = time.time()
            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_ctx": 8192},
            )
            latency_sec = time.time() - t0
            meta = {
                "latency_sec":        round(latency_sec, 3),
                "prompt_token_count": resp.prompt_eval_count,
                "output_token_count": resp.eval_count,
            }
            return resp.message.content, meta
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def _run_cell(
    model: str,
    technique: str,
    budget: int,
    counter: dict,
    data_dir: object,
    results_dir: object,
    runs_per_cell: int,
) -> None:
    """(model, technique, budget) 한 셀에 대해 runs_per_cell 개 샘플 실행."""
    template  = load_prompt_template(technique)
    schema_str = json.dumps(EXTRACTION_SCHEMA, ensure_ascii=False, indent=2)
    slug       = _model_slug(model)

    for idx in range(runs_per_cell):
        out_dir  = results_dir / slug / technique
        out_path = out_dir / f"result_{slug}_{technique}_{budget}_{idx}.json"
        if out_path.exists():
            counter["skipped"] += 1
            continue  # 재시작 시 중복 스킵

        budget_dir = data_dir / f"budget_{budget}" if data_dir != DATA_TEST_DIR else data_dir
        src_path = budget_dir / f"raw_{budget}_{idx}.json"
        if not src_path.exists():
            continue  # 데이터 없으면 조용히 넘어감

        with open(src_path, encoding="utf-8") as f:
            sample = json.load(f)

        prompt = template.format(
            document=sample["document"],
            schema=schema_str,
        )

        try:
            raw_output, meta = _call_model(model, prompt)
            error             = None
        except Exception as e:
            raw_output = ""
            meta       = {"latency_sec": None, "prompt_token_count": None, "output_token_count": None}
            error      = str(e)

        result = {
            "model":               model,
            "technique":           technique,
            "token_budget":        budget,
            "index":               idx,
            "input_document":      sample["document"],
            "answer":              sample["answer"],
            "raw_output":          raw_output,
            "error":               error,
            "latency_sec":         meta["latency_sec"],
            "prompt_token_count":  meta["prompt_token_count"],
            "output_token_count":  meta["output_token_count"],
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        if error is None:
            counter["completed"] += 1


def _write_experiment_log(
    started_at: datetime.datetime,
    counts: dict,
    models: list,
    techniques: list,
    budgets: list,
    runs_per_cell: int,
    test_mode: bool = False,
) -> None:
    finished_at = datetime.datetime.now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")
    log_prefix = "test_experiment" if test_mode else "experiment"

    log = {
        "run_id":      run_id,
        "stage":       "run_experiment",
        "test_mode":   test_mode,
        "started_at":  started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "parameters": {
            "models":        models,
            "techniques":    techniques,
            "token_budgets": budgets,
            "runs_per_cell": runs_per_cell,
            "temperature":   0.0,
        },
        "results_per_model": {
            model: {
                "completed": counts[model]["completed"],
                "skipped":   counts[model]["skipped"],
            }
            for model in models
        },
        "total": {
            "completed": sum(c["completed"] for c in counts.values()),
            "skipped":   sum(c["skipped"]   for c in counts.values()),
        },
    }

    logs_dir = TEST_LOGS_DIR if test_mode else LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{log_prefix}_{run_id}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"Log → {log_path}")


def run_all(test_mode: bool = False, only_techniques: list[str] | None = None) -> None:
    models        = TEST_CONFIG["models"]        if test_mode else OLLAMA_MODELS
    techniques    = TEST_CONFIG["techniques"]    if test_mode else PROMPTING_TECHNIQUES
    budgets       = TEST_CONFIG["token_budgets"] if test_mode else TOKEN_BUDGETS
    runs_per_cell = TEST_CONFIG["runs_per_cell"] if test_mode else RUNS_PER_CELL
    data_dir      = DATA_TEST_DIR                if test_mode else DATA_RAW_DIR
    results_dir   = TEST_EXPERIMENTS_DIR         if test_mode else EXPERIMENTS_DIR

    if only_techniques:
        techniques = [t for t in techniques if t in only_techniques]

    started_at = datetime.datetime.now()
    counts = {m: {"completed": 0, "skipped": 0} for m in models}

    total = len(models) * len(techniques) * len(budgets)
    with tqdm(total=total, desc="cells") as pbar:
        for model in models:
            for technique in techniques:
                for budget in budgets:
                    pbar.set_postfix(
                        model=_model_slug(model),
                        tech=technique,
                        budget=budget,
                    )
                    _run_cell(model, technique, budget, counts[model], data_dir, results_dir, runs_per_cell)
                    pbar.update(1)

    _write_experiment_log(started_at, counts, models, techniques, budgets, runs_per_cell, test_mode=test_mode)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--technique", nargs="+", help="실행할 기법만 지정 (예: --technique json_schema)")
    args = parser.parse_args()
    run_all(test_mode=args.test, only_techniques=args.technique)
