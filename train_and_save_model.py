from __future__ import annotations

"""Trening i zapis domyślnego modelu typu zdarzenia.

Uruchomienie tego skryptu tworzy plik `.joblib` w `models/`.
Uwaga: dotyczy tylko klasyfikatora typu zdarzenia (kategoria), a nie ekstrakcji relacji.
"""

import argparse
from pathlib import Path

from event_extractor import EventExtractor


def main() -> int:
    """CLI: trenuje model na danych i zapisuje go do pliku."""
    parser = argparse.ArgumentParser(description="Train event type model and save it.")
    parser.add_argument("--headlines", default="datasets/id_and_headline_first_sentence (1).csv")
    parser.add_argument("--tagged", default="datasets/tagged.csv")
    parser.add_argument("--out", default="models/event_type_model.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)

    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    extractor = EventExtractor()
    extractor.train(args.headlines, args.tagged, test_size=args.test_size)
    extractor.save_classifier(str(out_path))

    print(f"Saved model: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
