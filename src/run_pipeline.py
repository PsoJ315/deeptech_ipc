from __future__ import annotations

from pathlib import Path

from preprocess import load_and_preprocess_many, save_preprocessed
from term_extraction import (
    PatentTermExtractor,
    build_term_candidate_table,
    build_term_document_table,
    build_term_frequency_tables,
    build_term_occurrence_table,
    load_stopwords,
)


def main():
    project_root = Path(__file__).resolve().parent.parent

    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    tables_dir = project_root / "output" / "tables"
    config_dir = project_root / "config"

    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=== STEP 1: Preprocess ===")
    df = load_and_preprocess_many(raw_dir)
    save_preprocessed(df, processed_dir)

    print(f"[info] 전체 문서 수: {len(df):,}")
    print(f"[info] 연도 범위: {df['application_year'].min()} ~ {df['application_year'].max()}")

    general_stopwords = load_stopwords(config_dir / "stopwords.txt")
    domain_stopwords = load_stopwords(config_dir / "domain_stopwords.txt")

    print("=== STEP 2: Term Extraction ===")
    extractor = PatentTermExtractor(
        general_stopwords=general_stopwords,
        domain_stopwords=domain_stopwords,
        min_term_len=2,
        max_ngram_nouns=4,
    )

    occurrence_df = build_term_occurrence_table(df, extractor)
    document_df = build_term_document_table(df, extractor)

    occurrence_path = tables_dir / "term_occurrence_table.csv"
    document_path = tables_dir / "term_document_table.csv"

    occurrence_df.to_csv(occurrence_path, index=False, encoding="utf-8-sig")
    document_df.to_csv(document_path, index=False, encoding="utf-8-sig")

    print(f"[save] {occurrence_path}")
    print(f"[save] {document_path}")

    print("=== STEP 3: Frequency Tables ===")
    freq_tables = build_term_frequency_tables(occurrence_df, document_df)

    overall_path = tables_dir / "term_freq_overall.csv"
    year_path = tables_dir / "term_freq_by_year.csv"
    ipc_path = tables_dir / "term_freq_by_ipc.csv"

    freq_tables["overall"].to_csv(overall_path, index=False, encoding="utf-8-sig")
    freq_tables["by_year"].to_csv(year_path, index=False, encoding="utf-8-sig")
    freq_tables["by_ipc"].to_csv(ipc_path, index=False, encoding="utf-8-sig")

    print(f"[save] {overall_path}")
    print(f"[save] {year_path}")
    print(f"[save] {ipc_path}")

    print("=== STEP 4: Candidate Table ===")
    candidate_df = build_term_candidate_table(
        occurrence_df=occurrence_df,
        document_df=document_df,
        original_df=df,
        top_n_context=3,
    )
    candidate_path = tables_dir / "term_candidates.csv"
    candidate_df.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    print(f"[save] {candidate_path}")

    print("=== DONE ===")


if __name__ == "__main__":
    main()