from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class DatasetPaths:
    headlines_csv_path: str = "datasets/id_and_headline_first_sentence (1).csv"
    tagged_csv_path: str = "datasets/tagged.csv"


def _read_csv_with_fallback(
    path: str,
    *,
    sep: str,
    encodings: Iterable[str] = ("utf-8", "utf-8-sig", "cp1250"),
) -> pd.DataFrame:
    last_exc: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(
                path,
                sep=sep,
                encoding=enc,
                encoding_errors="replace",
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as exc:
            last_exc = exc
    raise last_exc  # type: ignore[misc]


def _normalize_label(raw: object) -> str:
    label = str(raw).strip()
    if "(" in label:
        label = label.split("(", 1)[0].strip()
    return label


def load_event_type_training_frame(
    *,
    headlines_csv_path: str = DatasetPaths.headlines_csv_path,
    tagged_csv_path: str = DatasetPaths.tagged_csv_path,
) -> pd.DataFrame:
    """Return DataFrame with columns: id, sentence, label.

    Joins:
    - headlines_csv_path: CSV with columns (id, headline)
    - tagged_csv_path: CSV with columns (id; kategoria; ...)
    """

    headlines = _read_csv_with_fallback(headlines_csv_path, sep=",")
    tagged = _read_csv_with_fallback(tagged_csv_path, sep=";")

    if "id" not in headlines.columns or "headline" not in headlines.columns:
        raise ValueError(
            f"Expected columns 'id' and 'headline' in {headlines_csv_path}, got: {list(headlines.columns)}"
        )
    if "id" not in tagged.columns or "kategoria" not in tagged.columns:
        raise ValueError(
            f"Expected columns 'id' and 'kategoria' in {tagged_csv_path}, got: {list(tagged.columns)}"
        )

    headlines = headlines[["id", "headline"]].copy()
    tagged = tagged[["id", "kategoria"]].copy()

    headlines["id"] = pd.to_numeric(headlines["id"], errors="coerce")
    tagged["id"] = pd.to_numeric(tagged["id"], errors="coerce")

    df = headlines.merge(tagged, on="id", how="inner")
    df = df.rename(columns={"headline": "sentence", "kategoria": "label"})

    df["sentence"] = df["sentence"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).map(_normalize_label)

    df = df.dropna(subset=["sentence", "label"])
    df = df[df["sentence"] != ""]
    df = df[df["label"] != ""]

    df = df.drop_duplicates(subset=["sentence", "label"]).reset_index(drop=True)
    return df


def load_relation_extraction_frame(
    *,
    headlines_csv_path: str = DatasetPaths.headlines_csv_path,
    tagged_csv_path: str = DatasetPaths.tagged_csv_path,
) -> pd.DataFrame:
    """Return DataFrame with sentence + gold KTO/CO/TRIGGER/GDZIE/KIEDY from tagged.csv.

    Columns returned:
    - id, sentence, label
    - gold_who, gold_what, gold_trigger, gold_where, gold_when
    """

    headlines = _read_csv_with_fallback(headlines_csv_path, sep=",")
    tagged = _read_csv_with_fallback(tagged_csv_path, sep=";")

    required_tagged = {"id", "kategoria", "KTO", "CO", "TRIGGER", "GDZIE", "KIEDY"}
    if "id" not in headlines.columns or "headline" not in headlines.columns:
        raise ValueError(
            f"Expected columns 'id' and 'headline' in {headlines_csv_path}, got: {list(headlines.columns)}"
        )
    missing = required_tagged.difference(set(tagged.columns))
    if missing:
        raise ValueError(
            f"Expected columns {sorted(required_tagged)} in {tagged_csv_path}, missing: {sorted(missing)}"
        )

    headlines = headlines[["id", "headline"]].copy()
    tagged = tagged[["id", "kategoria", "KTO", "CO", "TRIGGER", "GDZIE", "KIEDY"]].copy()

    headlines["id"] = pd.to_numeric(headlines["id"], errors="coerce")
    tagged["id"] = pd.to_numeric(tagged["id"], errors="coerce")

    df = headlines.merge(tagged, on="id", how="inner")
    df = df.rename(
        columns={
            "headline": "sentence",
            "kategoria": "label",
            "KTO": "gold_who",
            "CO": "gold_what",
            "TRIGGER": "gold_trigger",
            "GDZIE": "gold_where",
            "KIEDY": "gold_when",
        }
    )

    df["sentence"] = df["sentence"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).map(_normalize_label)
    for c in ["gold_who", "gold_what", "gold_trigger", "gold_where", "gold_when"]:
        df[c] = df[c].fillna("").astype(str).str.strip()

    df = df.dropna(subset=["sentence", "label"])
    df = df[df["sentence"] != ""]
    df = df[df["label"] != ""]

    df = df.reset_index(drop=True)
    return df
