from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

DATA_RAW_DIR  = BASE_DIR / "data" / "raw"   # 하위에 budget_{N}/ 서브폴더
DATA_TEST_DIR = BASE_DIR / "data" / "test"
PROMPTS_DIR   = BASE_DIR / "prompts"

EXPERIMENTS_DIR      = BASE_DIR / "experiments"       # 실험 원본 JSON
TEST_EXPERIMENTS_DIR = BASE_DIR / "experiments" / "test"

ANALYSIS_DIR      = BASE_DIR / "analysis"             # 채점·분석 결과
ANALYSIS_VERSION  = "v2"                              # 현재 채점 버전
TEST_ANALYSIS_DIR = BASE_DIR / "analysis" / "test"

LOGS_DIR      = BASE_DIR / "logs" / "production"
TEST_LOGS_DIR = BASE_DIR / "logs" / "test"

# ── Models ─────────────────────────────────────────────────────────────────────
OLLAMA_MODELS = [
    "qwen2.5:1.5b",
    "qwen2.5:3b",
    "qwen2.5:7b",
]

GPT_MODEL_GENERATE = "gpt-5.4-mini-2026-03-17"   # data generation
GPT_MODEL_EVAL     = "gpt-4o-mini"    # semantic scoring

# ── Prompting Techniques ───────────────────────────────────────────────────────
PROMPTING_TECHNIQUES = [
    "zero_shot",
    "few_shot",
    "cot",           # Chain-of-Thought
    "structured",
    "xml",
    "json_schema",
]

# ── Token Budget Tiers ─────────────────────────────────────────────────────────
TOKEN_BUDGETS = [200, 500, 1000, 2000, 4000]

# ── Experiment Settings ────────────────────────────────────────────────────────
RUNS_PER_CELL = 50  # repetitions per (model, technique, token_budget) cell
TOKEN_COUNT_TOLERANCE = 0.30  # actual_token_count 허용 범위 (±30%)

# ── Test Mode ──────────────────────────────────────────────────────────────────
TEST_CONFIG = {
    "models":             ["qwen2.5:1.5b"],
    "techniques":         ["zero_shot", "json_schema"],
    "token_budgets":      [200, 1000],
    "samples_per_budget": 2,
    "runs_per_cell":      2,
}

# ── JSON Schema (extraction target) ───────────────────────────────────────────
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "기관명":   {"type": "string", "description": "발신 기관 또는 담당 부서명"},
        "문서번호": {"type": "string", "description": "문서 고유 번호"},
        "작성일":   {"type": "string", "description": "작성일 (YYYY-MM-DD)"},
        "제목":     {"type": "string", "description": "문서 제목"},
        "핵심내용": {"type": "string", "description": "문서 핵심 내용 요약 (1-3문장)"},
        "담당자":   {"type": "string", "description": "담당자 이름 (직책 포함)"},
        "처리기한": {"type": "string", "description": "처리 또는 제출 기한 (YYYY-MM-DD)"},
    },
    "required": ["기관명", "문서번호", "작성일", "제목", "핵심내용", "담당자", "처리기한"],
}
