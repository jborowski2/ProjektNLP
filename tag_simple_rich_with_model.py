from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from data_loading import _read_csv_with_fallback
from event_extractor import EventExtractor


def _safe_write_csv(path: Path, df: pd.DataFrame, *, sep: str = ",") -> Path:
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
            "Tag sentences with the project's model (event type classifier + relation extractor) and write predictions to CSV."
        )
    )
    parser.add_argument(
        "--headlines",
        default="results/simple_rich_test_data_set_headlines.csv",
        help="CSV with columns: id,headline",
    )
    parser.add_argument(
        "--tagged",
        default=None,
        help="Optional tagged.csv-like file (semicolon-separated) to join gold columns for comparison.",
    )
    parser.add_argument(
        "--classifier",
        default="models/event_type_model.joblib",
        help="Path to trained event type classifier (.joblib).",
    )
    parser.add_argument(
        "--out",
        default="results/simple_rich_model_tagging.csv",
        help="Output predictions CSV.",
    )
    args = parser.parse_args()

    headlines = _read_csv_with_fallback(args.headlines, sep=",")
    if "id" not in headlines.columns or "headline" not in headlines.columns:
        raise ValueError(f"Expected columns id,headline in {args.headlines}, got {list(headlines.columns)}")

    headlines = headlines[["id", "headline"]].copy()
    headlines["id"] = pd.to_numeric(headlines["id"], errors="coerce")
    headlines = headlines.dropna(subset=["id", "headline"]).copy()
    headlines["headline"] = headlines["headline"].astype(str).str.strip()
    headlines = headlines[headlines["headline"] != ""].copy()

    extractor = EventExtractor()
    extractor.load_classifier(args.classifier)

    rows: list[dict[str, Any]] = []
    for id_val, sent in headlines.itertuples(index=False, name=None):
        event = extractor.extract_event(str(sent))
        rows.append(
            {
                "id": int(id_val),
                "sentence": str(sent),
                "pred_event_type": event.event_type,
                "pred_confidence": float(event.confidence),
                "pred_who": event.who or "",
                "pred_what": event.what or "",
                "pred_trigger": event.trigger or "",
                "pred_where": event.where or "",
                "pred_when": event.when or "",
            }
        )

    pred_df = pd.DataFrame(rows)

    if args.tagged:
        tagged = _read_csv_with_fallback(args.tagged, sep=";")
        if "id" not in tagged.columns:
            raise ValueError(f"Expected 'id' column in {args.tagged}, got {list(tagged.columns)}")
        tagged = tagged.copy()
        tagged["id"] = pd.to_numeric(tagged["id"], errors="coerce")
        tagged = tagged.dropna(subset=["id"]).copy()

        # Standard columns in tagged.csv
        rename_map = {
            "kategoria": "gold_label",
            "KTO": "gold_who",
            "CO": "gold_what",
            "TRIGGER": "gold_trigger",
            "GDZIE": "gold_where",
            "KIEDY": "gold_when",
        }
        for old, new in rename_map.items():
            if old in tagged.columns:
                tagged = tagged.rename(columns={old: new})

        keep = ["id"] + [c for c in rename_map.values() if c in tagged.columns]
        tagged = tagged[keep]
        pred_df = pred_df.merge(tagged, on="id", how="left")

    out_path = _safe_write_csv(Path(args.out), pred_df, sep=",")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
