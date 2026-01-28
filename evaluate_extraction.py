from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_loading import load_relation_extraction_frame
from relation_extractor import RelationExtractor


def _is_empty(x: str | None) -> bool:
    return x is None or str(x).strip() == ""


def _field_ok(
    rel: RelationExtractor,
    *,
    pred: str | None,
    gold: str | None,
    lenient_if_gold_empty: bool,
) -> bool:
    """Sentence-level correctness for one field.

    If lenient_if_gold_empty=True and gold is empty, we treat it as correct regardless of prediction.
    """
    pred_empty = _is_empty(pred)
    gold_empty = _is_empty(gold)

    if gold_empty:
        return True if lenient_if_gold_empty else pred_empty

    if pred_empty:
        return False

    return _match_by_lemmas(rel, str(pred), str(gold))


def _lemma_set(rel: RelationExtractor, text: str) -> set[str]:
    """Lemmas without function words; helps compare surface vs base forms (Polsce vs Polska)."""
    doc = rel.nlp(str(text))
    drop_pos = {"ADP", "DET", "CCONJ", "SCONJ", "PART", "PUNCT", "SPACE"}
    out: set[str] = set()
    for t in doc:
        if t.pos_ in drop_pos:
            continue
        lemma = (t.lemma_ or t.text).strip().lower()
        if not lemma:
            continue
        # Drop pure punctuation-ish leftovers
        if not re.search(r"\w", lemma, flags=re.UNICODE):
            continue
        out.add(lemma)
    return out


def _match_by_lemmas(rel: RelationExtractor, pred: str | None, gold: str | None) -> bool:
    if _is_empty(gold):
        return False
    if _is_empty(pred):
        return False
    gold_set = _lemma_set(rel, str(gold))
    pred_set = _lemma_set(rel, str(pred))
    if not gold_set or not pred_set:
        return False
    inter = gold_set.intersection(pred_set)
    # Require most of the gold content words to appear in prediction.
    # Be more tolerant for very short gold phrases.
    threshold = 0.8 if len(gold_set) >= 3 else 0.5
    return (len(inter) / len(gold_set)) >= threshold


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (prec == prec and rec == rec and (prec + rec)) else float("nan")
    return {"tp": float(tp), "fp": float(fp), "fn": float(fn), "precision": float(prec), "recall": float(rec), "f1": float(f1)}


def _field_prf_from_rows(rows: list[dict[str, Any]], *, field: str, rel: RelationExtractor) -> dict[str, float]:
    tp = fp = fn = 0
    pred_key = field
    gold_key = f"gold_{field}"

    strict_when = bool(rows[0].get("_strict_when")) if rows else True
    lenient_if_gold_empty = (field == "when") and (not strict_when)

    for r in rows:
        pred = r.get(pred_key)
        gold = r.get(gold_key)

        pred_empty = _is_empty(pred)
        gold_empty = _is_empty(gold)

        if (not pred_empty) and (not gold_empty) and _match_by_lemmas(rel, str(pred), str(gold)):
            tp += 1
        elif (not pred_empty) and gold_empty:
            # Lenient WHEN: don't penalize predictions when gold is empty.
            if not lenient_if_gold_empty:
                fp += 1
        elif pred_empty and (not gold_empty):
            fn += 1
        elif (not pred_empty) and (not gold_empty):
            fp += 1
            fn += 1

    return _prf(tp, fp, fn)


def _sentence_accuracy_from_rows(rows: list[dict[str, Any]], *, field: str, rel: RelationExtractor) -> float:
    pred_key = field
    gold_key = f"gold_{field}"
    ok = 0
    strict_when = bool(rows[0].get("_strict_when")) if rows else True
    lenient_if_gold_empty = (field == "when") and (not strict_when)
    for r in rows:
        pred = r.get(pred_key)
        gold = r.get(gold_key)

        if _field_ok(rel, pred=pred, gold=gold, lenient_if_gold_empty=lenient_if_gold_empty):
            ok += 1
    return float(ok / len(rows)) if rows else float("nan")


def _coverage_from_rows(rows: list[dict[str, Any]], *, field: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([1.0 if (not _is_empty(r.get(field))) else 0.0 for r in rows]))


def _gold_presence_from_rows(rows: list[dict[str, Any]], *, field: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([1.0 if (not _is_empty(r.get(f"gold_{field}"))) else 0.0 for r in rows]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate KTO/CO/GDZIE/KIEDY extraction on the full dataset (proxy metrics).")
    parser.add_argument("--headlines", default="datasets/id_and_headline_first_sentence (1).csv")
    parser.add_argument("--tagged", default="datasets/tagged.csv")
    parser.add_argument("--limit", type=int, default=0, help="Optional: limit number of examples for faster runs.")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument(
        "--proxy-ner",
        action="store_true",
        help="Use old NER-based proxy evaluation instead of gold KTO/CO/GDZIE/KIEDY from tagged.csv.",
    )
    parser.add_argument(
        "--strict-when",
        action="store_true",
        help="Strict WHEN evaluation: if gold KIEDY is empty, prediction must also be empty (otherwise it's counted as wrong).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="If >0, export top-N best-matching sentences (vs gold) and print metrics for that subset.",
    )
    parser.add_argument(
        "--top-out",
        default="results/top_matches.csv",
        help="Output CSV path for --top-n selection.",
    )
    parser.add_argument(
        "--top-corrected-tagged",
        default="results/tagged_top_corrected.csv",
        help=(
            "When used with --top-n, write a new tagged.csv-like file (id;kategoria;KTO;CO;TRIGGER;GDZIE;KIEDY) "
            "for the top subset, with suggested corrections based on predictions."
        ),
    )
    args = parser.parse_args()

    df = load_relation_extraction_frame(headlines_csv_path=args.headlines, tagged_csv_path=args.tagged)
    if args.limit and args.limit > 0 and args.limit < len(df):
        df = df.sample(n=int(args.limit), random_state=42).reset_index(drop=True)

    rel = RelationExtractor()

    # Counters (gold-based unless --proxy-ner)
    who_tp = who_fp = who_fn = 0
    what_tp = what_fp = what_fn = 0
    where_tp = where_fp = where_fn = 0
    when_tp = when_fp = when_fn = 0
    trigger_tp = trigger_fp = trigger_fn = 0

    who_correct = 0
    what_correct = 0
    where_correct = 0
    when_correct = 0
    trigger_correct = 0
    all5_correct = 0

    who_gold_present = 0
    what_gold_present = 0
    where_gold_present = 0
    when_gold_present = 0
    trigger_gold_present = 0

    rows: list[dict[str, Any]] = []

    for _, r in df.iterrows():
        sent = str(r["sentence"])
        doc = rel.nlp(sent)

        # Extraction (single parse)
        who, trigger, what, where, when = rel.extract_relations_from_doc(doc)

        gold_who = str(r.get("gold_who", "") or "").strip()
        gold_what = str(r.get("gold_what", "") or "").strip()
        gold_where = str(r.get("gold_where", "") or "").strip()
        gold_when = str(r.get("gold_when", "") or "").strip()
        gold_trigger = str(r.get("gold_trigger", "") or "").strip()

        if not args.proxy_ner:
            # Sentence-level correctness (treat empty/empty as correct).
            who_ok = _field_ok(rel, pred=who, gold=gold_who, lenient_if_gold_empty=False)
            what_ok = _field_ok(rel, pred=what, gold=gold_what, lenient_if_gold_empty=False)
            where_ok = _field_ok(rel, pred=where, gold=gold_where, lenient_if_gold_empty=False)
            when_ok = _field_ok(rel, pred=when, gold=gold_when, lenient_if_gold_empty=(not args.strict_when))
            trigger_ok = _field_ok(rel, pred=trigger, gold=gold_trigger, lenient_if_gold_empty=False)

            who_correct += 1 if who_ok else 0
            what_correct += 1 if what_ok else 0
            where_correct += 1 if where_ok else 0
            when_correct += 1 if when_ok else 0
            trigger_correct += 1 if trigger_ok else 0
            all5_correct += 1 if (who_ok and what_ok and where_ok and when_ok and trigger_ok) else 0

            # Gold presence
            who_gold_present += 0 if _is_empty(gold_who) else 1
            what_gold_present += 0 if _is_empty(gold_what) else 1
            where_gold_present += 0 if _is_empty(gold_where) else 1
            when_gold_present += 0 if _is_empty(gold_when) else 1
            trigger_gold_present += 0 if _is_empty(gold_trigger) else 1

            # WHO
            if not _is_empty(who) and not _is_empty(gold_who) and _match_by_lemmas(rel, who, gold_who):
                who_tp += 1
            elif not _is_empty(who) and _is_empty(gold_who):
                who_fp += 1
            elif _is_empty(who) and not _is_empty(gold_who):
                who_fn += 1
            elif (not _is_empty(who)) and (not _is_empty(gold_who)):
                who_fp += 1
                who_fn += 1

            # WHAT
            if not _is_empty(what) and not _is_empty(gold_what) and _match_by_lemmas(rel, what, gold_what):
                what_tp += 1
            elif not _is_empty(what) and _is_empty(gold_what):
                what_fp += 1
            elif _is_empty(what) and not _is_empty(gold_what):
                what_fn += 1
            elif (not _is_empty(what)) and (not _is_empty(gold_what)):
                what_fp += 1
                what_fn += 1

            # WHERE
            if not _is_empty(where) and not _is_empty(gold_where) and _match_by_lemmas(rel, where, gold_where):
                where_tp += 1
            elif not _is_empty(where) and _is_empty(gold_where):
                where_fp += 1
            elif _is_empty(where) and not _is_empty(gold_where):
                where_fn += 1
            elif (not _is_empty(where)) and (not _is_empty(gold_where)):
                where_fp += 1
                where_fn += 1

            # WHEN
            if not _is_empty(when) and not _is_empty(gold_when) and _match_by_lemmas(rel, when, gold_when):
                when_tp += 1
            elif not _is_empty(when) and _is_empty(gold_when):
                if args.strict_when:
                    when_fp += 1
            elif _is_empty(when) and not _is_empty(gold_when):
                when_fn += 1
            elif (not _is_empty(when)) and (not _is_empty(gold_when)):
                when_fp += 1
                when_fn += 1

            # TRIGGER (note: extractor uses lemma; compare by lemma)
            if not _is_empty(trigger) and not _is_empty(gold_trigger) and _match_by_lemmas(rel, trigger, gold_trigger):
                trigger_tp += 1
            elif not _is_empty(trigger) and _is_empty(gold_trigger):
                trigger_fp += 1
            elif _is_empty(trigger) and not _is_empty(gold_trigger):
                trigger_fn += 1
            elif (not _is_empty(trigger)) and (not _is_empty(gold_trigger)):
                trigger_fp += 1
                trigger_fn += 1
        else:
            # Proxy "ground truth" from NER and simple heuristics
            # pl_core_news_lg uses labels like: persName, geogName, placeName, date
            person_ents = [e.text for e in doc.ents if e.label_ in {"persName"}]
            loc_ents = [e.text for e in doc.ents if e.label_ in {"geogName", "placeName"}]
            time_ents = [e.text for e in doc.ents if e.label_ in {"date"}]

            has_person = len(person_ents) > 0
            has_loc = len(loc_ents) > 0
            has_time = len(time_ents) > 0

            def _overlap(haystack: str | None, needles: list[str]) -> bool:
                if not haystack:
                    return False
                h = haystack.lower()
                return any(n.lower() in h for n in needles)

            # WHO
            if who and has_person and _overlap(who, person_ents):
                who_tp += 1
            elif who and (not has_person):
                who_fp += 1
            elif (not who) and has_person:
                who_fn += 1

            # WHERE
            if where and has_loc and _overlap(where, loc_ents):
                where_tp += 1
            elif where and (not has_loc):
                where_fp += 1
            elif (not where) and has_loc:
                where_fn += 1

            # WHEN
            if when and has_time and _overlap(when, time_ents):
                when_tp += 1
            elif when and (not has_time):
                when_fp += 1
            elif (not when) and has_time:
                when_fn += 1

        rows.append(
            {
                "id": r.get("id"),
                "label": r.get("label"),
                "sentence": sent,
                "who": who,
                "what": what,
                "where": where,
                "when": when,
                "trigger": trigger,
                "gold_who": gold_who,
                "gold_what": gold_what,
                "gold_where": gold_where,
                "gold_when": gold_when,
                "gold_trigger": gold_trigger,
                "_ok_who": (None if args.proxy_ner else who_ok),
                "_ok_what": (None if args.proxy_ner else what_ok),
                "_ok_where": (None if args.proxy_ner else where_ok),
                "_ok_when": (None if args.proxy_ner else when_ok),
                "_ok_trigger": (None if args.proxy_ner else trigger_ok),
                "_strict_when": bool(args.strict_when),
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_path = out_dir / "extraction_predictions.csv"
    pd.DataFrame(rows).to_csv(detail_path, index=False, encoding="utf-8")

    summary = {
        "n": int(len(df)),
        "coverage": {
            "who_rate": float(np.mean([1.0 if x["who"] else 0.0 for x in rows])),
            "what_rate": float(np.mean([1.0 if x["what"] else 0.0 for x in rows])),
            "where_rate": float(np.mean([1.0 if x["where"] else 0.0 for x in rows])),
            "when_rate": float(np.mean([1.0 if x["when"] else 0.0 for x in rows])),
        },
        "sentence_accuracy": {
            "who": float(who_correct / int(len(df)) if len(df) else float("nan")),
            "what": float(what_correct / int(len(df)) if len(df) else float("nan")),
            "where": float(where_correct / int(len(df)) if len(df) else float("nan")),
            "when": float(when_correct / int(len(df)) if len(df) else float("nan")),
            "trigger": float(trigger_correct / int(len(df)) if len(df) else float("nan")),
            "all_fields": float(all5_correct / int(len(df)) if len(df) else float("nan")),
        },
        "gold_presence": {
            "who_rate": float(who_gold_present / int(len(df)) if len(df) else float("nan")),
            "what_rate": float(what_gold_present / int(len(df)) if len(df) else float("nan")),
            "where_rate": float(where_gold_present / int(len(df)) if len(df) else float("nan")),
            "when_rate": float(when_gold_present / int(len(df)) if len(df) else float("nan")),
            "trigger_rate": float(trigger_gold_present / int(len(df)) if len(df) else float("nan")),
        },
        "metrics": {
            "who": _prf(who_tp, who_fp, who_fn),
            "what": _prf(what_tp, what_fp, what_fn),
            "where": _prf(where_tp, where_fp, where_fn),
            "when": _prf(when_tp, when_fp, when_fn),
            "trigger": _prf(trigger_tp, trigger_fp, trigger_fn),
        },
        "notes": [
            "Domyślnie to są metryki względem ręcznych tagów z datasets/tagged.csv (kolumny KTO/CO/TRIGGER/GDZIE/KIEDY).",
            "Dopasowanie jest lemma-based (żeby np. 'w Polsce' pasowało do gold 'Polska').",
            "Uwaga: to nadal nie jest idealne dopasowanie spanów (różne zakresy fraz), ale jest dużo bliżej realnej oceny niż proxy NER.",
            "KIEDY: domyślnie ewaluacja jest łagodna – jeśli gold KIEDY jest puste, to uznajemy predykcję za poprawną (brak kary za FP). Użyj --strict-when żeby to zaostrzyć.",
            "Jeśli chcesz wrócić do starego trybu proxy przez NER: uruchom z --proxy-ner.",
        ],
    }

    summary_path = out_dir / "extraction_report.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {detail_path}")
    print(f"Wrote: {summary_path}")
    print("\n=== COVERAGE ===")
    for k, v in summary["coverage"].items():
        print(f"{k}: {v:.3f}")

    print("\n=== GOLD PRESENCE ===")
    for k, v in summary["gold_presence"].items():
        print(f"{k}: {v:.3f}")

    print("\n=== PRF vs GOLD (lemma-match) ===")
    for field in ["who", "what", "where", "when", "trigger"]:
        m = summary["metrics"][field]
        print(f"{field}: precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f} (tp={int(m['tp'])} fp={int(m['fp'])} fn={int(m['fn'])})")

    print("\n=== % ZDAŃ POPRAWNYCH (gold, lemma-match) ===")
    for field, v in summary["sentence_accuracy"].items():
        print(f"{field}: {v:.3f}")

    if (not args.proxy_ner) and args.top_n and args.top_n > 0:
        out_path = Path(args.top_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df_rows = pd.DataFrame(rows)
        # Score: number of correct fields among the 5
        score_cols = ["_ok_who", "_ok_what", "_ok_where", "_ok_when", "_ok_trigger"]
        df_rows["match_score"] = df_rows[score_cols].sum(axis=1)
        df_rows = df_rows.sort_values(by=["match_score"], ascending=False).head(int(args.top_n))

        export_cols = [
            "id",
            "label",
            "sentence",
            "trigger",
            "who",
            "what",
            "where",
            "when",
            "gold_trigger",
            "gold_who",
            "gold_what",
            "gold_where",
            "gold_when",
            "match_score",
        ]
        try:
            df_rows[export_cols].to_csv(out_path, index=False, encoding="utf-8")
        except PermissionError:
            alt = out_path.with_name(f"{out_path.stem}_alt{out_path.suffix}")
            df_rows[export_cols].to_csv(alt, index=False, encoding="utf-8")
            out_path = alt

        top_rows: list[dict[Any, Any]] = df_rows.to_dict(orient="records")
        print(f"\n=== TOP {int(args.top_n)} (best match_score) ===")
        print(f"Wrote: {out_path}")

        # Write a corrected tagged CSV for those rows.
        # Rule of thumb:
        # - If field was correct -> keep gold
        # - If field was incorrect and prediction is non-empty -> use prediction
        # - For KIEDY: if gold is empty, keep it empty (user requested lenient behavior)
        corrected_path = Path(args.top_corrected_tagged)
        corrected_path.parent.mkdir(parents=True, exist_ok=True)

        def _correct_field(row: dict[Any, Any], *, field: str) -> str:
            gold = str(row.get(f"gold_{field}", "") or "").strip()
            pred = str(row.get(field, "") or "").strip()
            ok = bool(row.get(f"_ok_{field}"))

            if field == "when" and _is_empty(gold):
                return ""

            if ok:
                return gold
            if not _is_empty(pred):
                return pred
            return gold

        corrected_df = pd.DataFrame(
            {
                "id": [r.get("id") for r in top_rows],
                "kategoria": [r.get("label") for r in top_rows],
                "KTO": [_correct_field(r, field="who") for r in top_rows],
                "CO": [_correct_field(r, field="what") for r in top_rows],
                "TRIGGER": [_correct_field(r, field="trigger") for r in top_rows],
                "GDZIE": [_correct_field(r, field="where") for r in top_rows],
                "KIEDY": [_correct_field(r, field="when") for r in top_rows],
            }
        )

        try:
            corrected_df.to_csv(corrected_path, index=False, sep=";", encoding="utf-8")
        except PermissionError:
            corrected_path = corrected_path.with_name(f"{corrected_path.stem}_alt{corrected_path.suffix}")
            corrected_df.to_csv(corrected_path, index=False, sep=";", encoding="utf-8")
        print(f"Wrote: {corrected_path}")

        # Print the same style metrics for the subset.
        print("\n=== TOP-N COVERAGE ===")
        for field in ["who", "what", "where", "when", "trigger"]:
            print(f"{field}_rate: {_coverage_from_rows(top_rows, field=field):.3f}")

        print("\n=== TOP-N GOLD PRESENCE ===")
        for field in ["who", "what", "where", "when", "trigger"]:
            print(f"{field}_rate: {_gold_presence_from_rows(top_rows, field=field):.3f}")

        print("\n=== TOP-N PRF vs GOLD (lemma-match) ===")
        for field in ["who", "what", "where", "when", "trigger"]:
            m = _field_prf_from_rows(top_rows, field=field, rel=rel)
            print(
                f"{field}: precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f} "
                f"(tp={int(m['tp'])} fp={int(m['fp'])} fn={int(m['fn'])})"
            )

        print("\n=== TOP-N % ZDAŃ POPRAWNYCH ===")
        for field in ["who", "what", "where", "when", "trigger"]:
            print(f"{field}: {_sentence_accuracy_from_rows(top_rows, field=field, rel=rel):.3f}")

        # All-fields correctness on the top-N
        all_ok = 0
        for r in top_rows:
            ok5 = True
            for field in ["who", "what", "where", "when", "trigger"]:
                pred = r.get(field)
                gold = r.get(f"gold_{field}")
                if _is_empty(pred) and _is_empty(gold):
                    continue
                if (not _is_empty(pred)) and (not _is_empty(gold)) and _match_by_lemmas(rel, str(pred), str(gold)):
                    continue
                ok5 = False
                break
            all_ok += 1 if ok5 else 0
        print(f"all_fields: {float(all_ok/len(top_rows)) if top_rows else float('nan'):.3f}")

        # Metrics vs corrected gold (should be higher by construction; useful for checking tag consistency).
        corrected_rows: list[dict[str, Any]] = []
        for r in top_rows:
            cr = dict(r)
            cr["gold_who"] = _correct_field(r, field="who")
            cr["gold_what"] = _correct_field(r, field="what")
            cr["gold_where"] = _correct_field(r, field="where")
            cr["gold_when"] = _correct_field(r, field="when")
            cr["gold_trigger"] = _correct_field(r, field="trigger")
            corrected_rows.append(cr)

        print("\n=== TOP-N METRICS vs CORRECTED TAGS ===")
        for field in ["who", "what", "where", "when", "trigger"]:
            m = _field_prf_from_rows(corrected_rows, field=field, rel=rel)
            print(
                f"{field}: precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f} "
                f"(tp={int(m['tp'])} fp={int(m['fp'])} fn={int(m['fn'])})"
            )

        all_ok2 = 0
        for r in corrected_rows:
            ok5 = True
            for field in ["who", "what", "where", "when", "trigger"]:
                pred = r.get(field)
                gold = r.get(f"gold_{field}")
                lenient = (field == "when")
                if _field_ok(rel, pred=pred, gold=gold, lenient_if_gold_empty=lenient):
                    continue
                ok5 = False
                break
            all_ok2 += 1 if ok5 else 0
        print(f"all_fields: {float(all_ok2/len(corrected_rows)) if corrected_rows else float('nan'):.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
