from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_loading import _read_csv_with_fallback
from relation_extractor import RelationExtractor


def _is_empty(x) -> bool:
    # Treat NaN as empty too.
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    return x is None or str(x).strip() == ""


def _lemma_set(rel: RelationExtractor, text: str) -> set[str]:
    doc = rel.nlp(str(text))
    drop_pos = {"ADP", "DET", "CCONJ", "SCONJ", "PART", "PUNCT", "SPACE"}
    out: set[str] = set()
    for t in doc:
        if t.pos_ in drop_pos:
            continue
        lemma = (t.lemma_ or t.text).strip().lower()
        if not lemma:
            continue
        if not any(ch.isalnum() for ch in lemma):
            continue
        out.add(lemma)
    return out


def _match_by_lemmas(rel: RelationExtractor, pred: str | None, gold: str | None) -> bool:
    if _is_empty(gold) or _is_empty(pred):
        return False
    gold_set = _lemma_set(rel, str(gold))
    pred_set = _lemma_set(rel, str(pred))
    if not gold_set or not pred_set:
        return False
    inter = gold_set.intersection(pred_set)
    threshold = 0.8 if len(gold_set) >= 3 else 0.5
    return (len(inter) / len(gold_set)) >= threshold


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (prec == prec and rec == rec and (prec + rec)) else float("nan")
    return float(prec), float(rec), float(f1)


def _field_prf(df: pd.DataFrame, *, field: str, rel: RelationExtractor) -> tuple[int, int, int, float, float, float]:
    pred_col = field
    gold_col = f"gold_{field}"
    tp = fp = fn = 0
    strict_when = bool(df.attrs.get("_strict_when", True))
    lenient_if_gold_empty = (field == "when") and (not strict_when)

    for _, r in df.iterrows():
        pred = r.get(pred_col)
        gold = r.get(gold_col)
        pred_empty = _is_empty(pred)
        gold_empty = _is_empty(gold)
        if (not pred_empty) and (not gold_empty) and _match_by_lemmas(rel, str(pred), str(gold)):
            tp += 1
        elif (not pred_empty) and gold_empty:
            if not lenient_if_gold_empty:
                fp += 1
        elif pred_empty and (not gold_empty):
            fn += 1
        elif (not pred_empty) and (not gold_empty):
            fp += 1
            fn += 1
    prec, rec, f1 = _prf(tp, fp, fn)
    return tp, fp, fn, prec, rec, f1


def _field_sentence_acc(df: pd.DataFrame, *, field: str, rel: RelationExtractor) -> float:
    pred_col = field
    gold_col = f"gold_{field}"
    ok = 0
    strict_when = bool(df.attrs.get("_strict_when", True))
    lenient_if_gold_empty = (field == "when") and (not strict_when)
    for _, r in df.iterrows():
        pred = r.get(pred_col)
        gold = r.get(gold_col)
        if _is_empty(gold):
            ok += 1 if (lenient_if_gold_empty or _is_empty(pred)) else 0
        else:
            ok += 1 if (not _is_empty(pred) and _match_by_lemmas(rel, str(pred), str(gold))) else 0
    return ok / len(df) if len(df) else float("nan")


def _safe_write(path: Path, df: pd.DataFrame, *, sep: str = ",") -> Path:
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
            "Compute extraction metrics for a preselected subset stored in results/top_matches.csv by running evaluate_extraction.py on that subset only."
        )
    )
    parser.add_argument("--top-matches", default="results/top_matches.csv")
    parser.add_argument(
        "--compute",
        action="store_true",
        help="Compute metrics directly from the columns in --top-matches (who/what/where/when/trigger + gold_*).",
    )
    parser.add_argument(
        "--strict-when",
        action="store_true",
        help="Strict WHEN: if gold_when is empty, prediction must also be empty (otherwise counts as FP / incorrect).",
    )
    parser.add_argument("--headlines-out", default="results/top_matches_headlines.csv")
    args = parser.parse_args()

    top = _read_csv_with_fallback(args.top_matches, sep=",")
    if "id" not in top.columns or "sentence" not in top.columns:
        raise ValueError(f"Expected columns id,sentence in {args.top_matches}, got {list(top.columns)}")

    headlines = top[["id", "sentence"]].rename(columns={"sentence": "headline"}).copy()
    headlines["id"] = pd.to_numeric(headlines["id"], errors="coerce")
    headlines = headlines.dropna(subset=["id", "headline"]).copy()
    headlines["id"] = headlines["id"].astype(int)
    headlines["headline"] = headlines["headline"].astype(str).str.strip()

    out_path = _safe_write(Path(args.headlines_out), headlines, sep=",")
    print(f"Wrote: {out_path}")

    if args.compute:
        required = {"who", "what", "where", "when", "trigger", "gold_who", "gold_what", "gold_where", "gold_when", "gold_trigger"}
        missing = required.difference(set(top.columns))
        if missing:
            raise ValueError(
                f"--compute requires columns {sorted(required)} in {args.top_matches}, missing: {sorted(missing)}"
            )
        rel = RelationExtractor()

        # Normalize NaNs to empty strings for stable emptiness checks + matching.
        top_norm = top.copy()
        for c in sorted(required):
            top_norm[c] = top_norm[c].fillna("").astype(str).str.strip()
        # Store evaluation mode on the DF for helper functions.
        top_norm.attrs["_strict_when"] = bool(args.strict_when)

        def rate_nonempty(col: str) -> float:
            return float(top_norm[col].ne("").mean())

        print("\n=== GOLD PRESENCE (from top_matches.csv) ===")
        for f in ["who", "what", "where", "when", "trigger"]:
            print(f"gold_{f}_rate: {rate_nonempty(f'gold_{f}'):.3f}")

        print("\n=== COVERAGE (pred presence) ===")
        for f in ["who", "what", "where", "when", "trigger"]:
            print(f"{f}_rate: {rate_nonempty(f):.3f}")

        print("\n=== PRF vs GOLD (lemma-match, from top_matches.csv) ===")
        for f in ["who", "what", "where", "when", "trigger"]:
            tp, fp, fn, prec, rec, f1 = _field_prf(top_norm, field=f, rel=rel)
            print(f"{f}: precision={prec:.3f} recall={rec:.3f} f1={f1:.3f} (tp={tp} fp={fp} fn={fn})")

        mode = "strict" if args.strict_when else "lenient_when"
        print(f"\n=== % ZDAŃ POPRAWNYCH ({mode}) ===")
        all_ok = 0
        for f in ["who", "what", "where", "when", "trigger"]:
            acc = _field_sentence_acc(top_norm, field=f, rel=rel)
            print(f"{f}: {acc:.3f}")

        for _, r in top_norm.iterrows():
            oks = []
            for f in ["who", "what", "where", "when", "trigger"]:
                pred = r.get(f)
                gold = r.get(f"gold_{f}")
                if _is_empty(gold):
                    if f == "when" and (not args.strict_when):
                        oks.append(True)
                    else:
                        oks.append(_is_empty(pred))
                else:
                    oks.append((not _is_empty(pred)) and _match_by_lemmas(rel, str(pred), str(gold)))
            all_ok += 1 if all(oks) else 0
        print(f"all_fields: {all_ok/len(top_norm):.3f}")

    else:
        print(
            "Now run (re-join gold from datasets/tagged.csv):\n"
            f"  .\\.venv\\Scripts\\python evaluate_extraction.py --headlines {out_path} --tagged datasets\\tagged.csv\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
