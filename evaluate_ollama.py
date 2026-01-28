from __future__ import annotations

"""Ewaluacja Qwen/Ollama na naszym zbiorze.

Liczy:
- klasyfikację typu zdarzenia vs gold `kategoria` (accuracy + F1),
- ekstrakcję relacji KTO/CO/TRIGGER/GDZIE/KIEDY vs gold w tagged.csv.

Uwaga: to jest ewaluacja "offline" na danych, ale wymaga uruchomionej Ollamy.
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data_loading import load_event_type_labels, load_relation_extraction_frame
from relation_extractor import RelationExtractor

from ollama_client import OllamaClient
from ollama_event_classifier import OllamaEventClassifier
from ollama_relation_extractor import OllamaRelationExtractor

# Reuse matching logic from evaluate_extraction.py to compare by lemmas.
import evaluate_extraction as ee


def _safe_json_loads(text: str):
    """Parse JSON nawet jeśli model dodał trochę tekstu dookoła."""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def _ollama_batch_predict(
    *,
    client: OllamaClient,
    model: str,
    allowed_labels: list[str],
    items: list[dict],
) -> list[dict]:
    """Zwróć listę wyników dla batcha.

    Każdy wynik ma klucze: idx, label, confidence, who, what, trigger, where, when.
    """

    allowed = ", ".join(sorted({str(x) for x in allowed_labels}))
    system = (
        "Jesteś systemem do ekstrakcji informacji z polskich zdań. "
        "Dla każdego zdania masz zwrócić: label (typ zdarzenia), confidence (0-1), "
        "who/what/trigger/where/when. "
        "Zwróć WYŁĄCZNIE poprawny JSON (bez komentarzy), jako listę obiektów. "
        "Nie zmyślaj: jeśli brak informacji, ustaw null. "
        "trigger zwróć najlepiej jako bezokolicznik."
    )
    user = (
        f"Dozwolone etykiety label: [{allowed}].\n"
        "Jeśli nie pasuje do żadnej, wybierz BRAK_ZDARZENIA (jeśli jest w słowniku).\n"
        "Wejście to lista obiektów {idx, sentence}.\n"
        "Wyjście ma być listą obiektów dokładnie w formacie:\n"
        "[{\"idx\": 0, \"label\": \"...\", \"confidence\": 0.0, "
        "\"who\": null, \"what\": null, \"trigger\": null, \"where\": null, \"when\": null}, ...]"
        "\n\n"
        f"Wejście: {json.dumps(items, ensure_ascii=False)}"
    )

    raw = client.chat(
        model=model,
        system=system,
        user=user,
        format="json",
        options={
            "temperature": 0.0,
            "top_p": 0.9,
            # batch może potrzebować więcej tokenów
            "num_predict": 1024,
        },
    )
    obj = _safe_json_loads(raw)
    if isinstance(obj, dict):
        if isinstance(obj.get("results"), list):
            obj = obj.get("results")
        elif isinstance(obj.get("items"), list):
            obj = obj.get("items")
        else:
            # single object fallback
            obj = [obj]
    if not isinstance(obj, list):
        raise ValueError("Expected JSON list")
    out: list[dict] = []
    for it in obj:
        if not isinstance(it, dict):
            continue
        out.append(it)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate Ollama (Qwen) on the project dataset")
    p.add_argument("--model", default="qwen2.5:7b-instruct")
    p.add_argument("--headlines", default="datasets/id_and_headline_first_sentence (1).csv")
    p.add_argument("--tagged", default="datasets/tagged.csv")
    p.add_argument("--limit", type=int, default=0, help="Optional: limit number of examples.")
    p.add_argument("--sample", action="store_true", help="If used with --limit, sample instead of taking head.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strict-when", action="store_true", help="Strict WHEN evaluation (same as evaluate_extraction.py).")
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Ile zdań na jedno wywołanie Ollamy (np. 5-10 jest zwykle dużo szybciej).",
    )
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_relation_extraction_frame(headlines_csv_path=args.headlines, tagged_csv_path=args.tagged)
    if args.limit and args.limit > 0 and args.limit < len(df):
        if args.sample:
            df = df.sample(n=int(args.limit), random_state=args.seed).reset_index(drop=True)
        else:
            df = df.head(int(args.limit)).reset_index(drop=True)

    batch_size = max(1, int(args.batch_size))
    clf = OllamaEventClassifier(model=args.model)
    rel_llm = OllamaRelationExtractor(model=args.model)
    client = OllamaClient(host="", timeout_s=300.0)
    allowed_labels = load_event_type_labels(tagged_csv_path=args.tagged)

    # For lemma-based matching we still use spaCy via RelationExtractor.
    rel_matcher = RelationExtractor()

    rows: list[dict] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    confs: list[float] = []

    t0 = time.time()
    df_rows = list(df.to_dict(orient="records"))
    n = len(df_rows)
    for start in range(0, n, batch_size):
        chunk = df_rows[start : start + batch_size]

        # batch request
        batch_items = [
            {"idx": int(start + i), "sentence": str(rr["sentence"])}
            for i, rr in enumerate(chunk)
        ]
        batch_results: dict[int, dict] = {}
        if batch_size > 1:
            try:
                pred_list = _ollama_batch_predict(
                    client=client,
                    model=args.model,
                    allowed_labels=allowed_labels,
                    items=batch_items,
                )
                for pr in pred_list:
                    try:
                        raw_idx = pr.get("idx")
                        if raw_idx is None:
                            continue
                        idx = int(raw_idx)
                    except Exception:
                        continue
                    batch_results[idx] = pr
            except Exception:
                batch_results = {}

        for i, r in enumerate(chunk):
            idx = int(start + i)
            sent = str(r["sentence"])
            gold_label = str(r.get("label", "") or "").strip()

            pr = batch_results.get(idx)
            if isinstance(pr, dict):
                pred_label = str(pr.get("label", "") or "").strip()
                try:
                    conf = float(pr.get("confidence", 0.0) or 0.0)
                except Exception:
                    conf = 0.0

                who = pr.get("who")
                what = pr.get("what")
                trigger = pr.get("trigger")
                where = pr.get("where")
                when = pr.get("when")

                # Normalize null-ish
                who = None if who in (None, "", "null") else str(who)
                what = None if what in (None, "", "null") else str(what)
                trigger = None if trigger in (None, "", "null") else str(trigger)
                where = None if where in (None, "", "null") else str(where)
                when = None if when in (None, "", "null") else str(when)
            else:
                # fallback: single calls (slower, but robust)
                pred_label, conf = clf.predict(sent)
                who, trigger, what, where, when = rel_llm.extract_relations(sent)

            y_true.append(gold_label)
            y_pred.append(pred_label)
            confs.append(float(conf))

            rows.append(
                {
                    "id": r.get("id"),
                    "sentence": sent,
                    "gold_label": gold_label,
                    "pred_label": pred_label,
                    "confidence": float(conf),
                    "gold_who": r.get("gold_who"),
                    "gold_what": r.get("gold_what"),
                    "gold_trigger": r.get("gold_trigger"),
                    "gold_where": r.get("gold_where"),
                    "gold_when": r.get("gold_when"),
                    "who": who,
                    "what": what,
                    "trigger": trigger,
                    "where": where,
                    "when": when,
                    "_strict_when": bool(args.strict_when),
                }
            )

        # Minimal progress
        done = min(start + batch_size, n)
        if done % max(1, min(50, batch_size * 5)) == 0 or done == n:
            print(f"Postęp: {done}/{n}")
    dt = time.time() - t0

    # Event type metrics
    acc = float(accuracy_score(y_true, y_pred)) if y_true else float("nan")
    f1_macro = float(f1_score(y_true, y_pred, average="macro")) if y_true else float("nan")
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted")) if y_true else float("nan")

    # Relations metrics (reuse evaluate_extraction functions)
    rel_fields = ["who", "what", "trigger", "where", "when"]
    rel_prf = {f: ee._field_prf_from_rows(rows, field=f, rel=rel_matcher) for f in rel_fields}
    rel_sent_acc = {f: ee._sentence_accuracy_from_rows(rows, field=f, rel=rel_matcher) for f in rel_fields}

    # All-5 sentence accuracy
    ok_all5 = 0
    for rr in rows:
        ok_who = ee._field_ok(rel_matcher, pred=rr.get("who"), gold=rr.get("gold_who"), lenient_if_gold_empty=False)
        ok_what = ee._field_ok(rel_matcher, pred=rr.get("what"), gold=rr.get("gold_what"), lenient_if_gold_empty=False)
        ok_where = ee._field_ok(rel_matcher, pred=rr.get("where"), gold=rr.get("gold_where"), lenient_if_gold_empty=False)
        ok_when = ee._field_ok(
            rel_matcher,
            pred=rr.get("when"),
            gold=rr.get("gold_when"),
            lenient_if_gold_empty=(not args.strict_when),
        )
        ok_trigger = ee._field_ok(rel_matcher, pred=rr.get("trigger"), gold=rr.get("gold_trigger"), lenient_if_gold_empty=False)
        ok_all5 += 1 if (ok_who and ok_what and ok_where and ok_when and ok_trigger) else 0
    all5_acc = float(ok_all5 / len(rows)) if rows else float("nan")

    report = {
        "model": args.model,
        "n": int(len(df)),
        "seconds": float(dt),
        "avg_seconds_per_sentence": float(dt / len(df)) if len(df) else float("nan"),
        "event_type": {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        },
        "relations": {
            "sentence_accuracy": rel_sent_acc,
            "prf": rel_prf,
            "all5_sentence_accuracy": all5_acc,
            "strict_when": bool(args.strict_when),
        },
    }

    # Save outputs
    safe_model = args.model.replace(":", "_").replace("/", "_")
    out_json = out_dir / f"ollama_eval_{safe_model}.json"
    out_csv = out_dir / f"ollama_predictions_{safe_model}.csv"

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print("=== OLLAMA EVAL ===")
    print(f"Model: {args.model}")
    print(f"N: {len(df)}  Czas: {dt:.1f}s  /zdanie: {report['avg_seconds_per_sentence']:.2f}s")
    print("\n--- Event type ---")
    print(f"Accuracy: {acc:.3f}  F1_macro: {f1_macro:.3f}  F1_weighted: {f1_weighted:.3f}")
    try:
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, zero_division=0))
    except Exception:
        pass

    print("\n--- Relations (lemma-match) ---")
    for f in rel_fields:
        prf = rel_prf[f]
        sa = rel_sent_acc[f]
        print(f"{f}: sent_acc={sa:.3f}  P={prf['precision']:.3f} R={prf['recall']:.3f} F1={prf['f1']:.3f}")
    print(f"ALL5 sent_acc={all5_acc:.3f}")
    print(f"\nZapisano: {out_json}  oraz  {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
