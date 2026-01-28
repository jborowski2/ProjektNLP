from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from data_loading import _read_csv_with_fallback
from relation_extractor import RelationExtractor


def _is_empty(x: object) -> bool:
    return x is None or str(x).strip() == ""


def _strip_leading_preps(text: str, *, preps: set[str]) -> str:
    s = text.strip()
    if not s:
        return s

    # Very lightweight: remove a single leading preposition token ("w Polsce" -> "Polsce").
    parts = s.split()
    if len(parts) >= 2 and parts[0].lower() in preps:
        return " ".join(parts[1:]).strip()
    return s


def _contains_case_insensitive(haystack: str, needle: str) -> bool:
    h = (haystack or "").lower()
    n = (needle or "").lower()
    return bool(n) and (n in h)


def _single_token_lemma(rel: RelationExtractor, text: str) -> str:
    s = text.strip()
    if not s:
        return s
    doc = rel.nlp(s)
    toks = [t for t in doc if (not t.is_space) and t.pos_ != "PUNCT"]
    if len(toks) != 1:
        return s
    t = toks[0]
    lemma = (t.lemma_ or t.text).strip()
    if not lemma:
        return s

    # Keep acronyms like "PKO" as-is.
    if s.isupper() and len(s) <= 6:
        return s

    # Months/days and proper nouns: normalize to lemma (e.g. "Holandii" -> "Holandia").
    if t.pos_ in {"PROPN", "NOUN", "ADJ"}:
        # Preserve capitalization pattern for proper nouns.
        if s[:1].isupper():
            return lemma[:1].upper() + lemma[1:]
        return lemma

    return s


def _normalize_field(rel: RelationExtractor, text: str, *, preps: set[str], lemma_single_token: bool) -> str:
    s = _strip_leading_preps(text, preps=preps)
    if lemma_single_token:
        s = _single_token_lemma(rel, s)
    return s.strip()


def _safe_to_csv(df: pd.DataFrame, path: Path, *, sep: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False, sep=sep, encoding="utf-8")
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_alt{path.suffix}")
        df.to_csv(alt, index=False, sep=sep, encoding="utf-8")
        return alt


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a corrected copy of tagged.csv using rule-based extraction (no classifier). "
            "Conservative policy: only fill empty fields; keep existing non-empty tags."
        )
    )
    parser.add_argument("--headlines", default="datasets/id_and_headline_first_sentence (1).csv")
    parser.add_argument("--tagged", default="datasets/tagged.csv")
    parser.add_argument("--out", default="results/tagged_corrected_by_rules_overwrite.csv")
    parser.add_argument("--review-out", default="results/tagged_corrections_review_overwrite.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Allow more overwrites (still rule-based, but less conservative).",
    )
    args = parser.parse_args()

    headlines = _read_csv_with_fallback(args.headlines, sep=",")
    tagged = _read_csv_with_fallback(args.tagged, sep=";")

    required_tagged = {"id", "kategoria", "KTO", "CO", "TRIGGER", "GDZIE", "KIEDY"}
    if "id" not in headlines.columns or "headline" not in headlines.columns:
        raise ValueError(f"Expected columns id,headline in {args.headlines}, got {list(headlines.columns)}")
    missing = required_tagged.difference(set(tagged.columns))
    if missing:
        raise ValueError(f"Missing columns in {args.tagged}: {sorted(missing)}")

    headlines = headlines[["id", "headline"]].copy()
    tagged = tagged[["id", "kategoria", "KTO", "CO", "TRIGGER", "GDZIE", "KIEDY"]].copy()

    headlines["id"] = pd.to_numeric(headlines["id"], errors="coerce")
    tagged["id"] = pd.to_numeric(tagged["id"], errors="coerce")

    df = headlines.merge(tagged, on="id", how="inner")
    df = df.rename(columns={"headline": "sentence"})
    df["sentence"] = df["sentence"].astype(str).str.strip()

    if args.limit and args.limit > 0 and args.limit < len(df):
        df = df.sample(n=int(args.limit), random_state=42).reset_index(drop=True)

    rel = RelationExtractor()

    # Preps for lightweight canonicalization.
    loc_preps = set(rel._location_preps)  # type: ignore[attr-defined]
    time_preps = set(rel._time_preps)  # type: ignore[attr-defined]

    out_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []

    sentences: list[str] = df["sentence"].tolist()
    docs = rel.nlp.pipe(sentences, batch_size=32)

    for row, doc in zip(df.to_dict(orient="records"), docs, strict=False):
        who_pred, _trigger_lemma, what_pred, where_pred, when_pred = rel.extract_relations_from_doc(doc)
        trigger_token = rel._pick_trigger_token(doc)  # type: ignore[attr-defined]
        trigger_pred = trigger_token.text if trigger_token is not None else ""

        # Normalize predicted strings to be closer to tagged style.
        where_norm = _strip_leading_preps(str(where_pred or ""), preps=loc_preps)
        when_norm = _strip_leading_preps(str(when_pred or ""), preps=time_preps)

        # Rule-based correction (high confidence):
        # - normalize obvious formatting issues (strip leading prepositions)
        # - normalize single-token inflections to lemma
        # - optionally overwrite when value doesn't occur in sentence and we have a strong alternative
        kto = str(row.get("KTO", "") or "").strip()
        co = str(row.get("CO", "") or "").strip()
        trig = str(row.get("TRIGGER", "") or "").strip()
        gdzie = str(row.get("GDZIE", "") or "").strip()
        kiedy = str(row.get("KIEDY", "") or "").strip()

        reasons: list[str] = []

        # Normalize (even if non-empty)
        kto_norm = _normalize_field(rel, kto, preps=loc_preps, lemma_single_token=True)
        if kto_norm != kto and not _is_empty(kto):
            reasons.append("KTO: normalize (strip-prep/lemma)")

        gdzie_norm2 = _normalize_field(rel, gdzie, preps=loc_preps, lemma_single_token=True)
        if gdzie_norm2 != gdzie and not _is_empty(gdzie):
            reasons.append("GDZIE: normalize (strip-prep/lemma)")

        kiedy_norm2 = _normalize_field(rel, kiedy, preps=time_preps, lemma_single_token=True)
        if kiedy_norm2 != kiedy and not _is_empty(kiedy):
            reasons.append("KIEDY: normalize (strip-prep/lemma)")

        # Trigger: if tag doesn't appear in sentence, replace with the best trigger token from the sentence.
        trig_new = trig
        if (not _is_empty(trig)) and (not _contains_case_insensitive(str(row.get("sentence") or ""), trig)):
            if not _is_empty(trigger_pred) and _contains_case_insensitive(str(row.get("sentence") or ""), trigger_pred):
                trig_new = str(trigger_pred).strip()
                reasons.append("TRIGGER: replace (tag not in sentence)")

        # For KTO/GDZIE/KIEDY, if existing value is empty, fill from extraction.
        kto_new = kto_norm
        if _is_empty(kto_new) and not _is_empty(who_pred):
            kto_new = _normalize_field(rel, str(who_pred), preps=loc_preps, lemma_single_token=True)
            reasons.append("KTO: fill empty")

        gdzie_new = gdzie_norm2
        if _is_empty(gdzie_new) and not _is_empty(where_norm):
            gdzie_new = _normalize_field(rel, str(where_norm), preps=loc_preps, lemma_single_token=True)
            reasons.append("GDZIE: fill empty")

        kiedy_new = kiedy_norm2
        if _is_empty(kiedy_new) and not _is_empty(when_norm):
            kiedy_new = _normalize_field(rel, str(when_norm), preps=time_preps, lemma_single_token=True)
            reasons.append("KIEDY: fill empty")

        # CO: generally harder; only fill empty, unless aggressive and the current CO doesn't occur in sentence.
        co_new = co
        if _is_empty(co_new) and not _is_empty(what_pred):
            co_new = str(what_pred).strip()
            reasons.append("CO: fill empty")
        elif args.aggressive and (not _is_empty(co_new)) and (not _contains_case_insensitive(str(row.get("sentence") or ""), co_new)):
            if not _is_empty(what_pred):
                co_new = str(what_pred).strip()
                reasons.append("CO: replace (tag not in sentence, aggressive)")

        # Extra high-confidence fix: adjective used as location (e.g. "turecki") -> use lemma of first LOC/GPE entity.
        if not _is_empty(gdzie_new):
            doc_g = rel.nlp(str(gdzie_new))
            toks = [t for t in doc_g if (not t.is_space) and t.pos_ != "PUNCT"]
            if len(toks) == 1 and toks[0].pos_ == "ADJ":
                loc_ents = [e for e in doc.ents if e.label_ in rel._ner_location_labels]  # type: ignore[attr-defined]
                if loc_ents:
                    repl = _single_token_lemma(rel, loc_ents[0].text)
                    if repl and repl != gdzie_new:
                        gdzie_new = repl
                        reasons.append("GDZIE: replace adjective with LOC entity")

        out_rows.append(
            {
                "id": row.get("id"),
                "kategoria": row.get("kategoria"),
                "KTO": kto_new,
                "CO": co_new,
                "TRIGGER": trig_new,
                "GDZIE": gdzie_new,
                "KIEDY": kiedy_new,
            }
        )

        # Review-friendly row with diffs.
        review_rows.append(
            {
                "id": row.get("id"),
                "kategoria": row.get("kategoria"),
                "sentence": row.get("sentence"),
                "reasons": "; ".join(reasons),
                "KTO_old": kto,
                "KTO_suggest": str(who_pred or "").strip(),
                "KTO_new": kto_new,
                "CO_old": co,
                "CO_suggest": str(what_pred or "").strip(),
                "CO_new": co_new,
                "TRIGGER_old": trig,
                "TRIGGER_suggest": str(trigger_pred or "").strip(),
                "TRIGGER_new": trig_new,
                "GDZIE_old": gdzie,
                "GDZIE_suggest": where_norm,
                "GDZIE_new": gdzie_new,
                "KIEDY_old": kiedy,
                "KIEDY_suggest": when_norm,
                "KIEDY_new": kiedy_new,
            }
        )

    out_path = _safe_to_csv(pd.DataFrame(out_rows), Path(args.out), sep=";")
    review_path = _safe_to_csv(pd.DataFrame(review_rows), Path(args.review_out), sep=",")

    print(f"Wrote: {out_path}")
    print(f"Wrote: {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
