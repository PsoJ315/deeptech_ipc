from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from kiwipiepy import Kiwi
from tqdm import tqdm


EN_TERM_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9\-\+_/]{1,}\b")
NUM_ONLY_PATTERN = re.compile(r"^\d+$")
MIXED_CLAIM_PATTERN = re.compile(r"^(제\d+|제\d+항|상기|청구항|실시예|단계)$")


def load_stopwords(stopword_path: str | Path) -> set[str]:
    path = Path(stopword_path)
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def normalize_term(term: str) -> str:
    term = term.strip()
    term = re.sub(r"\s+", " ", term)
    return term.lower()


class PatentTermExtractor:
    def __init__(
        self,
        general_stopwords: Iterable[str] | None = None,
        domain_stopwords: Iterable[str] | None = None,
        min_term_len: int = 2,
        max_ngram_nouns: int = 4,
    ):
        self.kiwi = Kiwi()
        self.general_stopwords = set(general_stopwords or [])
        self.domain_stopwords = set(domain_stopwords or [])
        self.stopwords = self.general_stopwords | self.domain_stopwords
        self.min_term_len = min_term_len
        self.max_ngram_nouns = max_ngram_nouns

    def _is_valid_term(self, term: str) -> bool:
        t = term.strip()
        if not t:
            return False
        if len(t) < self.min_term_len:
            return False
        if NUM_ONLY_PATTERN.match(t):
            return False
        if MIXED_CLAIM_PATTERN.match(t):
            return False
        if t in self.stopwords:
            return False
        if re.fullmatch(r"[^\w가-힣A-Za-z]+", t):
            return False
        return True

    def _extract_korean_compound_terms(self, text: str) -> list[str]:
        tokens = self.kiwi.tokenize(text)
        results: list[str] = []

        noun_buffer: list[str] = []

        def flush_buffer():
            nonlocal noun_buffer, results
            if not noun_buffer:
                return

            # 1-gram ~ max_ngram_nouns 생성
            n = len(noun_buffer)
            for size in range(1, min(self.max_ngram_nouns, n) + 1):
                for i in range(0, n - size + 1):
                    cand = "".join(noun_buffer[i:i + size])
                    if self._is_valid_term(cand):
                        results.append(cand)

                    cand_sp = " ".join(noun_buffer[i:i + size])
                    if self._is_valid_term(cand_sp):
                        results.append(cand_sp)

            noun_buffer = []

        for tok in tokens:
            form = tok.form.strip()
            tag = tok.tag

            # 일반 명사/고유명사/외국어/한자어류 위주
            if tag.startswith("NN") or tag in {"SL", "SH"}:
                noun_buffer.append(form)
            else:
                flush_buffer()

        flush_buffer()
        return results

    def _extract_english_terms(self, text: str) -> list[str]:
        terms = EN_TERM_PATTERN.findall(text)
        out = []
        for t in terms:
            nt = normalize_term(t)
            if self._is_valid_term(nt):
                out.append(nt)
        return out

    def extract_terms_from_text(self, text: str) -> list[str]:
        if not isinstance(text, str) or not text.strip():
            return []

        kor_terms = self._extract_korean_compound_terms(text)
        en_terms = self._extract_english_terms(text)

        merged = []
        seen = set()
        for t in kor_terms + en_terms:
            nt = normalize_term(t)
            if nt not in seen and self._is_valid_term(nt):
                seen.add(nt)
                merged.append(nt)

        return merged


def ensure_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return []
    return []


def build_term_doc_table(df: pd.DataFrame, extractor: PatentTermExtractor) -> pd.DataFrame:
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting terms"):
        terms = extractor.extract_terms_from_text(row["text"])
        ipc_prefixes = ensure_list(row["ipc_prefixes"])
        year = row.get("application_year", None)

        for term in terms:
            records.append({
                "application_no": row["application_no"],
                "term": term,
                "application_year": year,
                "ipc_prefixes": ipc_prefixes,
            })

    term_doc_df = pd.DataFrame(records)
    return term_doc_df


def build_term_frequency_tables(term_doc_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}

    if term_doc_df.empty:
        out["overall"] = pd.DataFrame(columns=["term", "tf", "df"])
        out["by_year"] = pd.DataFrame(columns=["application_year", "term", "tf", "df"])
        out["by_ipc"] = pd.DataFrame(columns=["ipc_prefix", "term", "tf", "df"])
        return out

    overall_tf = term_doc_df.groupby("term").size().reset_index(name="tf")
    overall_df = term_doc_df.groupby("term")["application_no"].nunique().reset_index(name="df")
    overall = overall_tf.merge(overall_df, on="term").sort_values(["tf", "df"], ascending=[False, False])

    out["overall"] = overall

    by_year_tf = term_doc_df.groupby(["application_year", "term"]).size().reset_index(name="tf")
    by_year_df = term_doc_df.groupby(["application_year", "term"])["application_no"].nunique().reset_index(name="df")
    by_year = by_year_tf.merge(by_year_df, on=["application_year", "term"])
    by_year = by_year.sort_values(["application_year", "tf"], ascending=[True, False])

    out["by_year"] = by_year

    exploded = term_doc_df.explode("ipc_prefixes").rename(columns={"ipc_prefixes": "ipc_prefix"})
    exploded = exploded.dropna(subset=["ipc_prefix"])

    by_ipc_tf = exploded.groupby(["ipc_prefix", "term"]).size().reset_index(name="tf")
    by_ipc_df = exploded.groupby(["ipc_prefix", "term"])["application_no"].nunique().reset_index(name="df")
    by_ipc = by_ipc_tf.merge(by_ipc_df, on=["ipc_prefix", "term"])
    by_ipc = by_ipc.sort_values(["ipc_prefix", "tf"], ascending=[True, False])

    out["by_ipc"] = by_ipc
    return out


def build_term_candidate_table(term_doc_df: pd.DataFrame, original_df: pd.DataFrame, top_n_context: int = 3) -> pd.DataFrame:
    if term_doc_df.empty:
        return pd.DataFrame(columns=["raw_term", "tf", "df", "sample_context", "top_ipc", "top_year"])

    tf_df = term_doc_df.groupby("term").size().reset_index(name="tf")
    df_df = term_doc_df.groupby("term")["application_no"].nunique().reset_index(name="df")

    ipc_counter = defaultdict(Counter)
    year_counter = defaultdict(Counter)

    for _, row in term_doc_df.iterrows():
        term = row["term"]
        year = row["application_year"]
        ipc_list = row["ipc_prefixes"] if isinstance(row["ipc_prefixes"], list) else []

        if pd.notna(year):
            year_counter[term][year] += 1
        for ipc in ipc_list:
            ipc_counter[term][ipc] += 1

    app_to_text = dict(zip(original_df["application_no"], original_df["text"]))
    term_to_apps = term_doc_df.groupby("term")["application_no"].apply(list).to_dict()

    records = []
    merged = tf_df.merge(df_df, on="term")

    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Building candidate table"):
        term = row["term"]
        apps = term_to_apps.get(term, [])
        contexts = []

        for app_no in apps[:top_n_context]:
            txt = app_to_text.get(app_no, "")
            idx = txt.lower().find(term.lower())
            if idx >= 0:
                start = max(0, idx - 40)
                end = min(len(txt), idx + len(term) + 80)
                contexts.append(txt[start:end])
            elif txt:
                contexts.append(txt[:120])

        top_ipc = ipc_counter[term].most_common(1)[0][0] if ipc_counter[term] else None
        top_year = year_counter[term].most_common(1)[0][0] if year_counter[term] else None

        records.append({
            "raw_term": term,
            "tf": row["tf"],
            "df": row["df"],
            "sample_context": " || ".join(contexts),
            "top_ipc": top_ipc,
            "top_year": top_year,
        })

    return pd.DataFrame(records).sort_values(["tf", "df"], ascending=[False, False])