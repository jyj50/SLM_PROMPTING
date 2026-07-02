"""prompts 패키지 — technique별 프롬프트 템플릿 로더."""

from pathlib import Path

_TEMPLATES: dict[str, str] = {}


def load_prompt_template(technique: str) -> str:
    """technique 이름으로 템플릿 문자열 반환. 없으면 KeyError."""
    if technique not in _TEMPLATES:
        path = Path(__file__).parent / f"{technique}.txt"
        if not path.exists():
            raise KeyError(f"Prompt template not found: {path}")
        _TEMPLATES[technique] = path.read_text(encoding="utf-8")
    return _TEMPLATES[technique]
