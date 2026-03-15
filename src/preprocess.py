from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pandas as pd


EXPECTED_COLUMNS = {
    "발명의명칭": "title",
    "요약": "abstract",
    "청구항": "claims",
    "IPC분류": "ipc_raw",
    "출원번호": "application_no",
    "출원일자": "application_date",
    "출원인": "applicant",
}

# IPC 형식이 붙어 있어도 prefix를 잡도록 경계조건을 완화
IPC_PREFIX_PATTERN = re.compile(r"([A-H]\d{2})[A-Z]?")
IPC_FULL_PATTERN = re.compile(r"([A-H]\d{2}[A-Z]?\s*\d+/\d+)")
YEAR_PATTERN = re.compile(r"(\d{4})")

CLAIM_CLEAN_PATTERNS = [
    r"\[?\s*청구항\s*\d+\s*\]?",
    r"제\s*\d+\s*항",
    r"제\s*\d+\s*항\s*내지\s*제\s*\d+\s*항",
    r"청구항\s*\d+",
    r"어느\s*한\s*항에\s*있어서",
    r"청구항에\s*있어서",
    r"상기",
    r"본\s*발명(?:은|에\s*따른)?",
    r"적어도\s*하나",
    r"복수의",
    r"일\s*실시예",
    r"실시예",
]


def load_kipris_csv(file_path: str | Path, skiprows: int = 7) -> pd.DataFrame:
    file_path = Path(file_path)

    tried = []
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(file_path, skiprows=skiprows, encoding=enc)
        except Exception as e:
            tried.append((enc, str(e)))

    msg = "\n".join([f"- {enc}: {err}" for enc, err in tried])
    raise ValueError(f"CSV 읽기 실패: {file_path}\n{msg}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}\n현재 컬럼: {list(df.columns)}")

    return df[list(EXPECTED_COLUMNS.keys())].rename(columns=EXPECTED_COLUMNS)


def clean_text(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_year(date_value: object) -> int | None:
    if pd.isna(date_value):
        return None
    s = str(date_value)
    m = YEAR_PATTERN.search(s)
    if not m:
        return None
    year = int(m.group(1))
    return year if 1900 <= year <= 2100 else None


def extract_ipc_prefixes(ipc_text: object) -> List[str]:
    if pd.isna(ipc_text):
        return []

    s = str(ipc_text).upper()
    s = s.replace(",", " ").replace(";", " ").replace("|", " ")
    s = re.sub(r"\s+", " ", s)

    matches = IPC_PREFIX_PATTERN.findall(s)
    return sorted(set(matches))


def extract_ipc_full_codes(ipc_text: object) -> List[str]:
    if pd.isna(ipc_text):
        return []

    s = str(ipc_text).upper()
    s = s.replace(",", " ").replace(";", " ").replace("|", " ")
    s = re.sub(r"\s+", " ", s)

    matches = IPC_FULL_PATTERN.findall(s)
    cleaned = [re.sub(r"\s+", "", x) for x in matches]
    return sorted(set(cleaned))


def strip_claim_boilerplate(text: object) -> str:
    if pd.isna(text):
        return ""
    out = str(text)

    for pat in CLAIM_CLEAN_PATTERNS:
        out = re.sub(pat, " ", out, flags=re.IGNORECASE)

    # 제1, 제2 같은 청구항 스타일 라벨 축약 제거
    out = re.sub(r"\b제\s*\d+\b", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def preprocess_patent_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df).copy()

    for col in ["title", "abstract", "claims", "ipc_raw", "application_no", "application_date", "applicant"]:
        df[col] = df[col].map(clean_text)

    df["application_year"] = df["application_date"].map(extract_year)
    df["ipc_prefixes"] = df["ipc_raw"].map(extract_ipc_prefixes)
    df["ipc_full_codes"] = df["ipc_raw"].map(extract_ipc_full_codes)

    # claims 쪽 boilerplate 제거를 먼저 적용
    df["claims_clean"] = df["claims"].map(strip_claim_boilerplate)

    df["text"] = (
        df["title"].fillna("") + " " +
        df["abstract"].fillna("") + " " +
        df["claims_clean"].fillna("")
    ).map(clean_text)

    df["text_len"] = df["text"].str.len()

    before = len(df)
    df = df.drop_duplicates(subset=["application_no"]).reset_index(drop=True)
    after = len(df)

    print(f"[preprocess] 중복 제거: {before - after}건")
    return df


def load_and_preprocess_many(raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    csv_files = sorted(raw_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"CSV 파일 없음: {raw_dir}")

    frames = []
    for fp in csv_files:
        print(f"[load] {fp.name}")
        raw_df = load_kipris_csv(fp)
        proc_df = preprocess_patent_df(raw_df)
        proc_df["source_file"] = fp.name
        frames.append(proc_df)

    merged = pd.concat(frames, ignore_index=True)

    before = len(merged)
    merged = merged.drop_duplicates(subset=["application_no"]).reset_index(drop=True)
    after = len(merged)
    print(f"[merge] 파일 간 중복 제거: {before - after}건")

    return merged


def save_preprocessed(df: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "patents_preprocessed.parquet"
    csv_path = output_dir / "patents_preprocessed.csv"

    try:
        df.to_parquet(parquet_path, index=False)
        print(f"[save] {parquet_path}")
    except Exception as e:
        print(f"[warn] parquet 저장 실패: {e}")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[save] {csv_path}")