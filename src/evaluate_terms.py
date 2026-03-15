from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


# =========================================
# Config
# =========================================

TOP_N_KEEP = 30
TOP_N_NOISE = 20
TOP_N_GENERIC = 20
TOP_N_BORDERLINE = 30
TOP_N_VARIANTS = 30
TOP_N_DOC_RATIO = 30


# =========================================
# Data classes
# =========================================

@dataclass
class EvalSummary:
    total_terms: int
    kept_terms: int
    dropped_terms: int
    keep_ratio: float
    patent_noise_count: int
    patent_noise_ratio: float
    generic_drop_count: int
    generic_drop_ratio: float
    generic_conditional_count: int
    generic_conditional_ratio: float
    generic_conditional_kept_count: int
    generic_conditional_kept_ratio_within_type: float
    df_eq_1_count: int
    df_eq_1_ratio: float
    high_doc_ratio_count: int
    high_doc_ratio_ratio: float
    avg_score: float
    median_score: float
    avg_variant_count: float
    median_variant_count: float
    fragmented_term_count: int
    fragmented_term_ratio: float


# =========================================
# Utilities
# =========================================

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def pct(n: int | float, d: int | float) -> float:
    if not d:
        return 0.0
    return round(float(n) / float(d), 4)


def to_builtin(x: Any) -> Any:
    if pd.isna(x):
        return None
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            return x
    return x


def df_to_records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    if limit is not None:
        df = df.head(limit)
    out = []
    for _, row in df.iterrows():
        out.append({k: to_builtin(v) for k, v in row.to_dict().items()})
    return out


def ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")


# =========================================
# Core evaluation
# =========================================

def load_inputs(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables_dir = project_root / "output" / "tables"

    review_df = safe_read_csv(tables_dir / "term_review_table.csv")
    overall_refined_df = safe_read_csv(tables_dir / "term_freq_overall_refined.csv")
    document_refined_df = safe_read_csv(tables_dir / "term_document_table_refined.csv")
    overall_raw_df = safe_read_csv(tables_dir / "term_freq_overall.csv")

    if review_df.empty:
        raise FileNotFoundError(
            f"{tables_dir / 'term_review_table.csv'} 가 없습니다. "
            "먼저 python src/run_pipeline.py 를 실행하세요."
        )

    required_review_cols = [
        "canonical_term", "representative_term", "tf", "df", "doc_ratio", "score",
        "term_type", "keep", "variant_count"
    ]
    ensure_columns(review_df, required_review_cols)

    review_df["keep"] = review_df["keep"].astype(bool)
    review_df["tf"] = pd.to_numeric(review_df["tf"], errors="coerce").fillna(0).astype(int)
    review_df["df"] = pd.to_numeric(review_df["df"], errors="coerce").fillna(0).astype(int)
    review_df["doc_ratio"] = pd.to_numeric(review_df["doc_ratio"], errors="coerce").fillna(0.0)
    review_df["score"] = pd.to_numeric(review_df["score"], errors="coerce").fillna(0.0)
    review_df["variant_count"] = pd.to_numeric(review_df["variant_count"], errors="coerce").fillna(0).astype(int)

    return review_df, overall_refined_df, document_refined_df, overall_raw_df


def build_summary(review_df: pd.DataFrame) -> EvalSummary:
    total_terms = len(review_df)
    kept_terms = int(review_df["keep"].sum())
    dropped_terms = total_terms - kept_terms

    patent_noise_count = int((review_df["term_type"] == "patent_noise").sum())
    generic_drop_count = int((review_df["term_type"] == "generic_drop").sum())
    generic_conditional_count = int((review_df["term_type"] == "generic_conditional").sum())
    generic_conditional_kept_count = int(
        ((review_df["term_type"] == "generic_conditional") & (review_df["keep"])).sum()
    )

    df_eq_1_count = int((review_df["df"] == 1).sum())
    high_doc_ratio_count = int((review_df["doc_ratio"] > 0.35).sum())
    fragmented_term_count = int((review_df["variant_count"] >= 2).sum())

    return EvalSummary(
        total_terms=total_terms,
        kept_terms=kept_terms,
        dropped_terms=dropped_terms,
        keep_ratio=pct(kept_terms, total_terms),
        patent_noise_count=patent_noise_count,
        patent_noise_ratio=pct(patent_noise_count, total_terms),
        generic_drop_count=generic_drop_count,
        generic_drop_ratio=pct(generic_drop_count, total_terms),
        generic_conditional_count=generic_conditional_count,
        generic_conditional_ratio=pct(generic_conditional_count, total_terms),
        generic_conditional_kept_count=generic_conditional_kept_count,
        generic_conditional_kept_ratio_within_type=pct(generic_conditional_kept_count, generic_conditional_count),
        df_eq_1_count=df_eq_1_count,
        df_eq_1_ratio=pct(df_eq_1_count, total_terms),
        high_doc_ratio_count=high_doc_ratio_count,
        high_doc_ratio_ratio=pct(high_doc_ratio_count, total_terms),
        avg_score=round(float(review_df["score"].mean()), 4),
        median_score=round(float(review_df["score"].median()), 4),
        avg_variant_count=round(float(review_df["variant_count"].mean()), 4),
        median_variant_count=round(float(review_df["variant_count"].median()), 4),
        fragmented_term_count=fragmented_term_count,
        fragmented_term_ratio=pct(fragmented_term_count, total_terms),
    )


def build_samples(review_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    keep_top = (
        review_df[review_df["keep"]]
        .sort_values(["score", "df", "tf"], ascending=[False, False, False])
        .head(TOP_N_KEEP)
        .copy()
    )

    patent_noise_top = (
        review_df[review_df["term_type"] == "patent_noise"]
        .sort_values(["df", "tf"], ascending=[False, False])
        .head(TOP_N_NOISE)
        .copy()
    )

    generic_drop_top = (
        review_df[review_df["term_type"] == "generic_drop"]
        .sort_values(["df", "tf"], ascending=[False, False])
        .head(TOP_N_GENERIC)
        .copy()
    )

    generic_conditional_dropped = (
        review_df[(review_df["term_type"] == "generic_conditional") & (~review_df["keep"])]
        .sort_values(["score", "df", "tf"], ascending=[False, False, False])
        .head(TOP_N_GENERIC)
        .copy()
    )

    borderline = (
        review_df[(review_df["term_type"] == "candidate") & (~review_df["keep"])]
        .sort_values(["score", "df", "tf"], ascending=[False, False, False])
        .head(TOP_N_BORDERLINE)
        .copy()
    )

    fragmented = (
        review_df[review_df["variant_count"] >= 2]
        .sort_values(["variant_count", "df", "tf"], ascending=[False, False, False])
        .head(TOP_N_VARIANTS)
        .copy()
    )

    broad_terms = (
        review_df.sort_values(["doc_ratio", "df", "tf"], ascending=[False, False, False])
        .head(TOP_N_DOC_RATIO)
        .copy()
    )

    return {
        "keep_top": keep_top,
        "patent_noise_top": patent_noise_top,
        "generic_drop_top": generic_drop_top,
        "generic_conditional_dropped": generic_conditional_dropped,
        "borderline": borderline,
        "fragmented": fragmented,
        "broad_terms": broad_terms,
    }


def flatten_samples(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for sample_type, df in samples.items():
        if df.empty:
            continue
        x = df.copy()
        x.insert(0, "sample_type", sample_type)
        frames.append(x)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def build_comparison(
    reports_dir: Path,
    current_summary: EvalSummary,
) -> dict[str, Any]:
    current_path = reports_dir / "term_evaluation_report.json"
    previous_path = reports_dir / "term_evaluation_report_prev.json"

    if current_path.exists():
        try:
            old = json.loads(current_path.read_text(encoding="utf-8"))
            previous_path.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    if not previous_path.exists():
        return {
            "has_previous": False,
            "message": "이전 평가 리포트가 없어 비교를 생략했습니다."
        }

    try:
        prev = json.loads(previous_path.read_text(encoding="utf-8"))
        prev_summary = prev.get("summary", {})
    except Exception:
        return {
            "has_previous": False,
            "message": "이전 평가 리포트를 읽지 못해 비교를 생략했습니다."
        }

    cur = asdict(current_summary)

    comparable_keys = [
        "total_terms",
        "kept_terms",
        "keep_ratio",
        "patent_noise_count",
        "patent_noise_ratio",
        "generic_drop_count",
        "generic_drop_ratio",
        "generic_conditional_count",
        "generic_conditional_ratio",
        "generic_conditional_kept_count",
        "generic_conditional_kept_ratio_within_type",
        "df_eq_1_count",
        "df_eq_1_ratio",
        "high_doc_ratio_count",
        "high_doc_ratio_ratio",
        "avg_score",
        "median_score",
        "avg_variant_count",
        "median_variant_count",
        "fragmented_term_count",
        "fragmented_term_ratio",
    ]

    deltas = {}
    for k in comparable_keys:
        prev_val = prev_summary.get(k)
        cur_val = cur.get(k)
        if prev_val is None or cur_val is None:
            continue
        try:
            deltas[k] = round(float(cur_val) - float(prev_val), 4)
        except Exception:
            continue

    return {
        "has_previous": True,
        "message": "직전 평가 결과와 비교했습니다.",
        "delta": deltas,
    }


# =========================================
# Report writers
# =========================================

def write_json_report(
    output_path: Path,
    summary: EvalSummary,
    samples: dict[str, pd.DataFrame],
    comparison: dict[str, Any],
    extra_meta: dict[str, Any],
) -> None:
    payload = {
        "summary": asdict(summary),
        "comparison_to_previous": comparison,
        "meta": extra_meta,
        "samples": {
            k: df_to_records(v)
            for k, v in samples.items()
        },
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_markdown_report(
    output_path: Path,
    summary: EvalSummary,
    samples: dict[str, pd.DataFrame],
    comparison: dict[str, Any],
    extra_meta: dict[str, Any],
) -> None:
    lines: list[str] = []

    lines.append("# Term Evaluation Report")
    lines.append("")
    lines.append("## 개요")
    lines.append(f"- total_terms: {summary.total_terms}")
    lines.append(f"- kept_terms: {summary.kept_terms}")
    lines.append(f"- dropped_terms: {summary.dropped_terms}")
    lines.append(f"- keep_ratio: {summary.keep_ratio:.2%}")
    lines.append("")

    lines.append("## 핵심 품질 지표")
    lines.append(f"- patent_noise_count: {summary.patent_noise_count} ({summary.patent_noise_ratio:.2%})")
    lines.append(f"- generic_drop_count: {summary.generic_drop_count} ({summary.generic_drop_ratio:.2%})")
    lines.append(f"- generic_conditional_count: {summary.generic_conditional_count} ({summary.generic_conditional_ratio:.2%})")
    lines.append(
        f"- generic_conditional_kept_count: {summary.generic_conditional_kept_count} "
        f"({summary.generic_conditional_kept_ratio_within_type:.2%} within type)"
    )
    lines.append(f"- df_eq_1_count: {summary.df_eq_1_count} ({summary.df_eq_1_ratio:.2%})")
    lines.append(f"- high_doc_ratio_count: {summary.high_doc_ratio_count} ({summary.high_doc_ratio_ratio:.2%})")
    lines.append(f"- fragmented_term_count: {summary.fragmented_term_count} ({summary.fragmented_term_ratio:.2%})")
    lines.append(f"- avg_score: {summary.avg_score}")
    lines.append(f"- median_score: {summary.median_score}")
    lines.append(f"- avg_variant_count: {summary.avg_variant_count}")
    lines.append(f"- median_variant_count: {summary.median_variant_count}")
    lines.append("")

    lines.append("## 참고 메타")
    for k, v in extra_meta.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## 직전 버전 비교")
    lines.append(f"- has_previous: {comparison.get('has_previous', False)}")
    lines.append(f"- message: {comparison.get('message', '')}")
    if comparison.get("has_previous"):
        delta = comparison.get("delta", {})
        for k, v in delta.items():
            sign = "+" if v > 0 else ""
            lines.append(f"- {k}: {sign}{v}")
    lines.append("")

    def add_term_list(title: str, df: pd.DataFrame, cols: list[str]) -> None:
        lines.append(f"## {title}")
        if df.empty:
            lines.append("- 데이터 없음")
            lines.append("")
            return

        for _, row in df.iterrows():
            parts = []
            for col in cols:
                if col in row.index:
                    parts.append(f"{col}={row[col]}")
            lines.append(f"- " + ", ".join(parts))
        lines.append("")

    add_term_list(
        "상위 유지 후보",
        samples["keep_top"],
        ["representative_term", "score", "df", "tf", "top_ipc", "top_year"],
    )
    add_term_list(
        "제거된 특허 문체 노이즈",
        samples["patent_noise_top"],
        ["representative_term", "df", "tf", "variants"],
    )
    add_term_list(
        "제거된 범용어",
        samples["generic_drop_top"],
        ["representative_term", "score", "df", "tf"],
    )
    add_term_list(
        "문맥 부족으로 탈락한 조건부 범용어",
        samples["generic_conditional_dropped"],
        ["representative_term", "score", "df", "tf", "top_context_terms"],
    )
    add_term_list(
        "경계선 후보",
        samples["borderline"],
        ["representative_term", "score", "df", "tf", "top_context_terms"],
    )
    add_term_list(
        "표기 파편화가 큰 용어",
        samples["fragmented"],
        ["representative_term", "variant_count", "df", "tf", "variants"],
    )
    add_term_list(
        "문서 범위가 지나치게 넓은 용어",
        samples["broad_terms"],
        ["representative_term", "doc_ratio", "df", "tf", "term_type"],
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


# =========================================
# Main
# =========================================

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    reports_dir = project_root / "output" / "reports"
    tables_dir = project_root / "output" / "tables"

    reports_dir.mkdir(parents=True, exist_ok=True)

    review_df, overall_refined_df, document_refined_df, overall_raw_df = load_inputs(project_root)

    summary = build_summary(review_df)
    samples = build_samples(review_df)
    comparison = build_comparison(reports_dir, summary)

    extra_meta = {
        "tables_dir": str(tables_dir),
        "review_table_exists": (tables_dir / "term_review_table.csv").exists(),
        "overall_refined_exists": (tables_dir / "term_freq_overall_refined.csv").exists(),
        "document_refined_exists": (tables_dir / "term_document_table_refined.csv").exists(),
        "overall_raw_exists": (tables_dir / "term_freq_overall.csv").exists(),
        "overall_refined_rows": int(len(overall_refined_df)) if not overall_refined_df.empty else 0,
        "document_refined_rows": int(len(document_refined_df)) if not document_refined_df.empty else 0,
        "overall_raw_rows": int(len(overall_raw_df)) if not overall_raw_df.empty else 0,
    }

    json_path = reports_dir / "term_evaluation_report.json"
    md_path = reports_dir / "term_evaluation_report.md"
    sample_csv_path = reports_dir / "term_eval_samples.csv"

    write_json_report(json_path, summary, samples, comparison, extra_meta)
    write_markdown_report(md_path, summary, samples, comparison, extra_meta)

    sample_df = flatten_samples(samples)
    sample_df.to_csv(sample_csv_path, index=False, encoding="utf-8-sig")

    print("=== TERM EVALUATION DONE ===")
    print(f"[save] {json_path}")
    print(f"[save] {md_path}")
    print(f"[save] {sample_csv_path}")


if __name__ == "__main__":
    main()