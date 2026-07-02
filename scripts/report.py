"""
report.py
─────────
eval_results.csv를 읽어 플롯 6개 생성.

출력:
  plot_valid_json_rate.png    — JSON 유효율 (±1 std 밴드)
  plot_schema_valid_rate.png  — Schema 준수율 (±1 std 밴드)
  plot_semantic_score.png     — 의미적 정확도 (±1 std 밴드)
  plot_latency.png            — token budget별 추론 시간
  plot_field_bar.png          — 기법별 필드 점수 bar chart (모델별 서브플롯)
  plot_field_line.png         — 필드별 token_budget × technique 라인 플롯 (필드×모델 그리드)
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ANALYSIS_DIR, ANALYSIS_VERSION, TEST_ANALYSIS_DIR,
    OLLAMA_MODELS, PROMPTING_TECHNIQUES, TOKEN_BUDGETS,
)

# 한국어 폰트 설정 (Malgun Gothic 우선)
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    if _font in _available_fonts:
        matplotlib.rcParams["font.family"] = _font
        break
matplotlib.rcParams["axes.unicode_minus"] = False

# 소프트 쿨톤 파스텔 색상 — technique별 고정
_COLORS = {
    "zero_shot":   "#7BB8D4",  # 소프트 블루
    "few_shot":    "#6BBFB5",  # 소프트 틸
    "cot":         "#9BA8D0",  # 소프트 인디고
    "structured":  "#B8A8D4",  # 소프트 퍼플
    "xml":         "#6EC6D8",  # 소프트 사이언
    "json_schema": "#A8C4A0",  # 소프트 그린
}

_MODEL_LABELS = {
    "qwen2.5:1.5b": "Qwen2.5 1.5B",
    "qwen2.5:3b":   "Qwen2.5 3B",
    "qwen2.5:7b":   "Qwen2.5 7B",
}

_FIELD_COLS = [
    "score_기관명", "score_문서번호", "score_작성일",
    "score_제목", "score_핵심내용", "score_담당자", "score_처리기한",
]
_FIELD_LABELS = ["기관명", "문서번호", "작성일", "제목", "핵심내용", "담당자", "처리기한"]

_PLOTS = [
    ("is_valid_json",  "JSON 유효율",   "plot_valid_json_rate.png"),
    ("schema_valid",   "Schema 준수율", "plot_schema_valid_rate.png"),
    ("semantic_score", "의미적 정확도", "plot_semantic_score.png"),
]


def _load(results_dir) -> pd.DataFrame:
    path = results_dir / "eval_results.csv"
    if not path.exists():
        raise FileNotFoundError("eval_results.csv 없음. evaluate.py를 먼저 실행하세요.")
    return pd.read_csv(path)


def _plot_metric(df: pd.DataFrame, metric: str, y_label: str, out_name: str, models, techniques, budgets, results_dir) -> None:
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n + 2.8, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    legend_handles, legend_labels = [], []

    for ax, model in zip(axes, models):
        for technique in techniques:
            sub = df[(df["model"] == model) & (df["technique"] == technique)]
            if sub.empty or sub[metric].isna().all():
                continue

            color = _COLORS.get(technique, "#AAAAAA")
            grp_mean = sub.groupby("token_budget")[metric].mean()
            grp_std  = sub.groupby("token_budget")[metric].std().fillna(0)

            (line,) = ax.plot(
                grp_mean.index,
                grp_mean.values,
                marker="o",
                markersize=5,
                linewidth=1.8,
                color=color,
                label=technique,
            )
            # ±1 std shaded region (rate 메트릭은 [0, 1]로 클리핑)
            ax.fill_between(
                grp_mean.index,
                (grp_mean - grp_std).clip(lower=0),
                (grp_mean + grp_std).clip(upper=1),
                alpha=0.12,
                color=color,
                linewidth=0,
            )
            # 범례용 핸들은 첫 번째 서브플롯 기준으로 한 번만 수집
            if model == models[0]:
                legend_handles.append(line)
                legend_labels.append(technique)

        ax.set_title(_MODEL_LABELS.get(model, model), fontsize=11, pad=8)
        ax.set_xlabel("Token Budget", fontsize=9)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets], fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel(y_label, fontsize=10)

    # 범례: 마지막 서브플롯 오른쪽 바깥
    fig.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9,
        frameon=False,
        title="Technique",
        title_fontsize=9,
    )

    fig.suptitle(y_label, fontsize=13, y=1.03)
    fig.tight_layout()

    out_path = results_dir / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def _plot_latency(df: pd.DataFrame, models, techniques, budgets, results_dir) -> None:
    """token_budget별 평균 추론 시간 라인 플롯 (모델별 서브플롯, technique별 선)."""
    if "latency_sec" not in df.columns or df["latency_sec"].isna().all():
        print("latency_sec 데이터 없음, 스킵.")
        return

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n + 2.8, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    legend_handles, legend_labels = [], []

    for ax, model in zip(axes, models):
        for technique in techniques:
            sub = df[(df["model"] == model) & (df["technique"] == technique)]
            if sub.empty or sub["latency_sec"].isna().all():
                continue

            color     = _COLORS.get(technique, "#AAAAAA")
            grp_mean  = sub.groupby("token_budget")["latency_sec"].mean()
            grp_std   = sub.groupby("token_budget")["latency_sec"].std().fillna(0)

            (line,) = ax.plot(
                grp_mean.index,
                grp_mean.values,
                marker="o",
                markersize=5,
                linewidth=1.8,
                color=color,
                label=technique,
            )
            ax.fill_between(
                grp_mean.index,
                (grp_mean - grp_std).clip(lower=0),
                grp_mean + grp_std,
                alpha=0.12,
                color=color,
                linewidth=0,
            )
            if model == models[0]:
                legend_handles.append(line)
                legend_labels.append(technique)

        ax.set_title(_MODEL_LABELS.get(model, model), fontsize=11, pad=8)
        ax.set_xlabel("Token Budget", fontsize=9)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets], fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("추론 시간 (초)", fontsize=10)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9,
        frameon=False,
        title="Technique",
        title_fontsize=9,
    )
    fig.suptitle("Token Budget별 추론 시간", fontsize=13, y=1.03)
    fig.tight_layout()

    out_path = results_dir / "plot_latency.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def _plot_field_bar(df: pd.DataFrame, models, techniques, results_dir) -> None:
    """기법별 필드 점수 bar chart. 모델별 서브플롯, x축=필드, 색상=technique."""
    field_cols = [c for c in _FIELD_COLS if c in df.columns]
    if not field_cols:
        print("필드 점수 컬럼 없음, bar chart 스킵.")
        return

    labels = [c.replace("score_", "") for c in field_cols]
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n + 2.8, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    bar_width = 0.12
    x = range(len(field_cols))
    legend_handles, legend_labels = [], []

    for ax, model in zip(axes, models):
        for i, technique in enumerate(techniques):
            sub = df[(df["model"] == model) & (df["technique"] == technique)]
            if sub.empty:
                continue
            means = [sub[c].mean() if c in sub.columns else 0 for c in field_cols]
            offset = (i - len(techniques) / 2 + 0.5) * bar_width
            bars = ax.bar(
                [xi + offset for xi in x],
                means,
                width=bar_width,
                color=_COLORS.get(technique, "#AAAAAA"),
                label=technique,
                alpha=0.85,
            )
            if model == models[0]:
                legend_handles.append(bars[0])
                legend_labels.append(technique)

        ax.set_title(_MODEL_LABELS.get(model, model), fontsize=11, pad=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, alpha=0.25, linestyle="--", axis="y")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("평균 점수", fontsize=10)
    fig.legend(legend_handles, legend_labels, loc="center left",
               bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False,
               title="Technique", title_fontsize=9)
    fig.suptitle("기법별 필드 점수", fontsize=13, y=1.03)
    fig.tight_layout()

    out_path = results_dir / "plot_field_bar.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def _plot_field_line(df: pd.DataFrame, models, techniques, budgets, results_dir) -> None:
    """필드별 token_budget × technique 라인 플롯. 행=필드, 열=모델 그리드."""
    field_cols = [c for c in _FIELD_COLS if c in df.columns]
    if not field_cols:
        print("필드 점수 컬럼 없음, line chart 스킵.")
        return

    n_fields = len(field_cols)
    n_models = len(models)
    fig, axes = plt.subplots(
        n_fields, n_models,
        figsize=(4.2 * n_models + 1.5, 2.8 * n_fields),
        sharey=True, sharex=True,
    )
    # axes를 항상 2D로 처리
    if n_fields == 1:
        axes = [axes]
    if n_models == 1:
        axes = [[ax] for ax in axes]

    for row, (col, field_label) in enumerate(zip(field_cols, _FIELD_LABELS)):
        for col_idx, model in enumerate(models):
            ax = axes[row][col_idx]
            for technique in techniques:
                sub = df[(df["model"] == model) & (df["technique"] == technique)]
                if sub.empty or col not in sub.columns or sub[col].isna().all():
                    continue
                grp = sub.groupby("token_budget")[col].mean()
                ax.plot(grp.index, grp.values, marker="o", markersize=3,
                        linewidth=1.4, color=_COLORS.get(technique, "#AAAAAA"),
                        label=technique)

            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)

            if row == 0:
                ax.set_title(_MODEL_LABELS.get(model, model), fontsize=9, pad=6)
            if col_idx == 0:
                ax.set_ylabel(field_label, fontsize=8, rotation=0,
                              labelpad=40, va="center")
            if row == n_fields - 1:
                ax.set_xlabel("Token Budget", fontsize=8)
                ax.set_xticks(budgets)
                ax.set_xticklabels([str(b) for b in budgets], fontsize=6, rotation=30)

    # 공통 범례
    handles = [plt.Line2D([0], [0], color=_COLORS.get(t, "#AAAAAA"),
                          linewidth=1.8, label=t) for t in techniques]
    fig.legend(handles, techniques, loc="center left",
               bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False,
               title="Technique", title_fontsize=8)

    fig.suptitle("필드별 Token Budget × 점수", fontsize=13, y=1.01)
    fig.tight_layout()

    out_path = results_dir / "plot_field_line.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def generate_report(test_mode: bool = False) -> None:
    results_dir = TEST_ANALYSIS_DIR if test_mode else ANALYSIS_DIR / ANALYSIS_VERSION
    df = _load(results_dir)

    models     = sorted(df["model"].unique())
    techniques = sorted(df["technique"].unique())
    budgets    = sorted(df["token_budget"].unique())

    for metric, y_label, out_name in _PLOTS:
        if metric == "semantic_score" and df["semantic_score"].isna().all():
            print("semantic_score 데이터 없음 (GPT 채점 미실행), 스킵.")
            continue
        _plot_metric(df, metric, y_label, out_name, models, techniques, budgets, results_dir)

    _plot_latency(df, models, techniques, budgets, results_dir)
    _plot_field_bar(df, models, techniques, results_dir)
    _plot_field_line(df, models, techniques, budgets, results_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    generate_report(test_mode=args.test)
