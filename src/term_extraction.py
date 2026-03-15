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
KOR_ONLY_PATTERN = re.compile(r"^[가-힣]+$")
PUNCT_ONLY_PATTERN = re.compile(r"^[^\w가-힣A-Za-z]+$")

MIXED_CLAIM_PATTERN = re.compile(
    r"^(제\d+|제\d+항|청구항|청구|상기|실시예|실시 예|단계|항|항중|항 중)$"
)

SPACE_NORMALIZATION_RULES = {
    "청구 항": "청구항",
    "항 중": "항중",
    "실시 예": "실시예",
    "제조 방법": "제조방법",
    "제어 방법": "제어방법",
    "분석 방법": "분석방법",
    "처리 방법": "처리방법",
    "조성 물": "조성물",
    "화합 물": "화합물",
    "반도 체": "반도체",
    "배터 리": "배터리",
    "센 서": "센서",
    "신경 망": "신경망",
    "인공 신경망": "인공신경망",
    "딥 러닝": "딥러닝",
    "머신 러닝": "머신러닝",
    "데이터 베이스": "데이터베이스",
    "전 해질": "전해질",
    "고체 전해질": "고체전해질",
    "전고체 전지": "전고체전지",
    "연료 전지": "연료전지",
    "태양 전지": "태양전지",
    "이차 전지": "이차전지",
    "반응 기": "반응기",
    "열 교환기": "열교환기",
}

COMPOUND_JOIN_PATTERNS = [
    re.compile(r"^([가-힣]{1,12})\s(센서|소자|재료|장치|모듈|배터리|전지|전극|전해질|반도체|기판|필름|코팅|촉매|조성물|화합물|수지|섬유|기기|회로|칩|패키지|항체|단백질|유전자|세포|약물|진단)$"),
    re.compile(r"^([가-힣]{1,12})\s(방법|공정|구조|패턴|네트워크|시스템|플랫폼|알고리즘|모델|엔진)$"),
]

ALLOWED_SHORT_KOR_TERMS = {
    "전극", "센서", "기판", "회로", "촉매", "항체", "세포", "전지",
    "양극", "음극", "광학", "영상", "전자", "회로", "칩", "레이더",
    "배터리", "반도체", "전해질", "신호", "소자", "촉매", "섬유",
}

GENERIC_SINGLE_NOUNS = {
    "금속", "물질", "장치", "방법", "정보", "데이터", "시스템", "모듈",
    "구성", "연결", "형성", "제어", "특징", "영역", "위치", "부분",
    "신호", "사용", "이용", "처리", "생성", "제공", "복수", "이상",
    "이하", "가능", "기준", "상태", "입력", "출력", "내부", "외부",
    "상부", "하부", "전기", "활성", "구조", "부재",
}


def load_stopwords(stopword_path: str | Path) -> set[str]:
    path = Path(stopword_path)
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


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


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_term(term: str) -> str:
    term = term.strip().lower()
    term = normalize_whitespace(term)

    term = re.sub(r"\s*-\s*", "-", term)
    term = re.sub(r"\s*/\s*", "/", term)
    term = re.sub(r"\s*\+\s*", "+", term)

    for src, dst in SPACE_NORMALIZATION_RULES.items():
        term = term.replace(src.lower(), dst.lower())

    for pattern in COMPOUND_JOIN_PATTERNS:
        m = pattern.match(term)
        if m:
            term = f"{m.group(1)}{m.group(2)}"

    return term


class PatentTermExtractor:
    def __init__(
        self,
        general_stopwords: Iterable[str] | None = None,
        domain_stopwords: Iterable[str] | None = None,
        min_term_len: int = 2,
        max_ngram_nouns: int = 4,
        min_kor_single_len: int = 3,
        min_en_term_len: int = 2,
    ):
        self.kiwi = Kiwi()
        self.general_stopwords = set(general_stopwords or [])
        self.domain_stopwords = set(domain_stopwords or [])
        self.stopwords = {normalize_term(x) for x in (self.general_stopwords | self.domain_stopwords)}
        self.min_term_len = min_term_len
        self.max_ngram_nouns = max_ngram_nouns
        self.min_kor_single_len = min_kor_single_len
        self.min_en_term_len = min_en_term_len

    def _is_stopword(self, term: str) -> bool:
        return normalize_term(term) in self.stopwords

    def _is_valid_term(self, term: str) -> bool:
        t = normalize_term(term)

        if not t:
            return False
        if len(t) < self.min_term_len:
            return False
        if NUM_ONLY_PATTERN.match(t):
            return False
        if PUNCT_ONLY_PATTERN.match(t):
            return False
        if MIXED_CLAIM_PATTERN.match(t):
            return False
        if self._is_stopword(t):
            return False

        if re.fullmatch(r"[a-z0-9\-\+_/]+", t):
            if len(t) < self.min_en_term_len:
                return False

        if KOR_ONLY_PATTERN.match(t) and " " not in t:
            if t in GENERIC_SINGLE_NOUNS:
                return False
            if len(t) < self.min_kor_single_len and t not in ALLOWED_SHORT_KOR_TERMS:
                return False

        if re.match(r"^\d+[a-z가-힣]*$", t):
            return False

        t2 = re.sub(r"(들|등)$", "", t)
        if not t2:
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

            n = len(noun_buffer)

            # 복합어를 우선 수집
            for size in range(min(self.max_ngram_nouns, n), 1, -1):
                for i in range(0, n - size + 1):
                    parts = noun_buffer[i:i + size]

                    cand_join = normalize_term("".join(parts))
                    if self._is_valid_term(cand_join):
                        results.append(cand_join)

                    cand_space = normalize_term(" ".join(parts))
                    if self._is_valid_term(cand_space):
                        results.append(cand_space)

            # 단일어는 더 엄격하게 선별
            for part in noun_buffer:
                cand = normalize_term(part)
                if self._is_valid_term(cand):
                    results.append(cand)

            noun_buffer = []

        for tok in tokens:
            form = tok.form.strip()
            tag = tok.tag

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
            if nt in {"or", "and", "of", "for", "to", "in", "on", "by", "at", "a", "an", "the"}:
                continue
            if self._is_valid_term(nt):
                out.append(nt)

        return out

    def _suppress_component_unigrams(self, terms: list[str]) -> list[str]:
        """
        복합어가 있는 경우 그 구성 단일어를 억제.
        예: 전기활성물질, 활성물질이 있으면 전기/활성/물질은 약화 또는 제거.
        현재는 보수적으로 제거.
        """
        complex_terms = [t for t in terms if (" " in t) or (KOR_ONLY_PATTERN.match(t) and len(t) >= 4)]
        single_terms = [t for t in terms if t not in complex_terms]

        blocked = set()
        for ct in complex_terms:
            # 띄어쓴 복합어
            if " " in ct:
                parts = ct.split()
                for p in parts:
                    if KOR_ONLY_PATTERN.match(p):
                        blocked.add(p)
            # 붙여쓴 복합어에서 자주 나타나는 기술 접미 기준 느슨한 분해
            for suffix in [
                "센서", "소자", "재료", "장치", "모듈", "배터리", "전지", "전극", "전해질",
                "반도체", "기판", "필름", "코팅", "촉매", "조성물", "화합물", "수지", "섬유",
                "기기", "회로", "칩", "패키지", "항체", "단백질", "유전자", "세포", "약물",
                "진단", "방법", "공정", "구조", "패턴", "네트워크", "시스템", "플랫폼",
                "알고리즘", "모델", "엔진"
            ]:
                if ct.endswith(suffix):
                    stem = ct[:-len(suffix)]
                    if KOR_ONLY_PATTERN.match(stem) and stem:
                        blocked.add(stem)
                    blocked.add(suffix)

        out = []
        for t in terms:
            if t in blocked and t not in ALLOWED_SHORT_KOR_TERMS:
                continue
            out.append(t)
        return out

    def extract_terms_from_text(self, text: str, deduplicate_within_doc: bool = False) -> list[str]:
        if not isinstance(text, str) or not text.strip():
            return []

        kor_terms = self._extract_korean_compound_terms(text)
        en_terms = self._extract_english_terms(text)

        merged = [normalize_term(t) for t in (kor_terms + en_terms) if self._is_valid_term(t)]
        merged = self._suppress_component_unigrams(merged)

        if deduplicate_within_doc:
            deduped = []
            seen = set()
            for t in merged:
                if t not in seen:
                    seen.add(t)
                    deduped.append(t)
            return deduped

        return merged


def build_term_occurrence_table(df: pd.DataFrame, extractor: PatentTermExtractor) -> pd.DataFrame:
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting term occurrences"):
        terms = extractor.extract_terms_from_text(row["text"], deduplicate_within_doc=False)
        ipc_prefixes = ensure_list(row.get("ipc_prefixes", []))
        year = row.get("application_year", None)

        for term in terms:
            records.append({
                "application_no": row["application_no"],
                "term": term,
                "application_year": year,
                "ipc_prefixes": ipc_prefixes,
            })

    return pd.DataFrame(records)


def build_term_document_table(df: pd.DataFrame, extractor: PatentTermExtractor) -> pd.DataFrame:
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting document terms"):
        terms = extractor.extract_terms_from_text(row["text"], deduplicate_within_doc=True)
        ipc_prefixes = ensure_list(row.get("ipc_prefixes", []))
        year = row.get("application_year", None)

        for term in terms:
            records.append({
                "application_no": row["application_no"],
                "term": term,
                "application_year": year,
                "ipc_prefixes": ipc_prefixes,
            })

    return pd.DataFrame(records)


def build_term_frequency_tables(
    occurrence_df: pd.DataFrame,
    document_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    out = {}

    if occurrence_df.empty or document_df.empty:
        out["overall"] = pd.DataFrame(columns=["term", "tf", "df"])
        out["by_year"] = pd.DataFrame(columns=["application_year", "term", "tf", "df"])
        out["by_ipc"] = pd.DataFrame(columns=["ipc_prefix", "term", "tf", "df"])
        return out

    overall_tf = occurrence_df.groupby("term").size().reset_index(name="tf")
    overall_df = document_df.groupby("term")["application_no"].nunique().reset_index(name="df")
    overall = overall_tf.merge(overall_df, on="term", how="outer").fillna(0)
    overall["tf"] = overall["tf"].astype(int)
    overall["df"] = overall["df"].astype(int)
    overall = overall.sort_values(["tf", "df", "term"], ascending=[False, False, True])
    out["overall"] = overall

    by_year_tf = occurrence_df.groupby(["application_year", "term"]).size().reset_index(name="tf")
    by_year_df = document_df.groupby(["application_year", "term"])["application_no"].nunique().reset_index(name="df")
    by_year = by_year_tf.merge(by_year_df, on=["application_year", "term"], how="outer").fillna(0)
    by_year["tf"] = by_year["tf"].astype(int)
    by_year["df"] = by_year["df"].astype(int)
    by_year = by_year.sort_values(["application_year", "tf", "df"], ascending=[True, False, False])
    out["by_year"] = by_year

    occ_ipc = occurrence_df.explode("ipc_prefixes").rename(columns={"ipc_prefixes": "ipc_prefix"})
    occ_ipc = occ_ipc.dropna(subset=["ipc_prefix"])

    doc_ipc = document_df.explode("ipc_prefixes").rename(columns={"ipc_prefixes": "ipc_prefix"})
    doc_ipc = doc_ipc.dropna(subset=["ipc_prefix"])

    by_ipc_tf = occ_ipc.groupby(["ipc_prefix", "term"]).size().reset_index(name="tf")
    by_ipc_df = doc_ipc.groupby(["ipc_prefix", "term"])["application_no"].nunique().reset_index(name="df")
    by_ipc = by_ipc_tf.merge(by_ipc_df, on=["ipc_prefix", "term"], how="outer").fillna(0)
    by_ipc["tf"] = by_ipc["tf"].astype(int)
    by_ipc["df"] = by_ipc["df"].astype(int)
    by_ipc = by_ipc.sort_values(["ipc_prefix", "tf", "df"], ascending=[True, False, False])
    out["by_ipc"] = by_ipc

    return out


def build_term_candidate_table(
    occurrence_df: pd.DataFrame,
    document_df: pd.DataFrame,
    original_df: pd.DataFrame,
    top_n_context: int = 3,
) -> pd.DataFrame:
    if occurrence_df.empty or document_df.empty:
        return pd.DataFrame(columns=["raw_term", "tf", "df", "doc_ratio", "sample_context", "top_ipc", "top_year"])

    tf_df = occurrence_df.groupby("term").size().reset_index(name="tf")
    df_df = document_df.groupby("term")["application_no"].nunique().reset_index(name="df")

    total_docs = max(1, original_df["application_no"].nunique())

    ipc_counter = defaultdict(Counter)
    year_counter = defaultdict(Counter)

    for _, row in document_df.iterrows():
        term = row["term"]
        year = row["application_year"]
        ipc_list = row["ipc_prefixes"] if isinstance(row["ipc_prefixes"], list) else []

        if pd.notna(year):
            year_counter[term][year] += 1
        for ipc in ipc_list:
            ipc_counter[term][ipc] += 1

    app_to_text = dict(zip(original_df["application_no"], original_df["text"]))
    term_to_apps = document_df.groupby("term")["application_no"].apply(list).to_dict()

    merged = tf_df.merge(df_df, on="term", how="outer").fillna(0)
    merged["tf"] = merged["tf"].astype(int)
    merged["df"] = merged["df"].astype(int)
    merged["doc_ratio"] = merged["df"] / total_docs

    records = []
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Building candidate table"):
        term = row["term"]
        apps = term_to_apps.get(term, [])
        contexts = []

        for app_no in apps[:top_n_context]:
            txt = app_to_text.get(app_no, "")
            txt_low = txt.lower()
            idx = txt_low.find(term.lower())

            if idx >= 0:
                start = max(0, idx - 50)
                end = min(len(txt), idx + len(term) + 100)
                contexts.append(txt[start:end])
            elif txt:
                contexts.append(txt[:150])

        top_ipc = ipc_counter[term].most_common(1)[0][0] if ipc_counter[term] else None
        top_year = year_counter[term].most_common(1)[0][0] if year_counter[term] else None

        records.append({
            "raw_term": term,
            "tf": int(row["tf"]),
            "df": int(row["df"]),
            "doc_ratio": float(row["doc_ratio"]),
            "sample_context": " || ".join(contexts),
            "top_ipc": top_ipc,
            "top_year": top_year,
        })

    candidate_df = pd.DataFrame(records)
    candidate_df = candidate_df.sort_values(["tf", "df", "raw_term"], ascending=[False, False, True])
    return candidate_df