"""Orkiestrator: klasyfikacja typu + ekstrakcja relacji.

`EventExtractor` scala dwa niezależne komponenty:
- `EventClassifier`: przewiduje typ zdarzenia (kategoria),
- `RelationExtractor`: heurystycznie wyciąga KTO/CO/TRIGGER/GDZIE/KIEDY.

W UI i skryptach to jest główny punkt wejścia do analizy pojedynczego zdania.
"""

import pandas as pd
from typing import Optional, Protocol, runtime_checkable
from event_record import EventRecord
from event_classifier import EventClassifier
from relation_extractor import RelationExtractor
from data_loading import load_event_type_training_frame


@runtime_checkable
class EventTypeClassifier(Protocol):
    def predict(self, sentence: str) -> tuple[str, float]:
        ...


@runtime_checkable
class TrainableEventTypeClassifier(EventTypeClassifier, Protocol):
    def train(self, sentences, labels, *args, **kwargs):
        ...


@runtime_checkable
class RelationExtractionModel(Protocol):
    def extract_relations(self, sentence: str) -> tuple[
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
        Optional[str],
    ]:
        ...


class EventExtractor:
    """Wysokopoziomowa analiza zdania (typ + relacje)."""

    def __init__(
        self,
        *,
        classifier: EventTypeClassifier | None = None,
        relations: RelationExtractionModel | None = None,
    ):
        self.classifier = classifier or EventClassifier()
        self.relations = relations or RelationExtractor()

    def train(
        self,
        headlines_csv_path: str = "datasets/id_and_headline_first_sentence (1).csv",
        tagged_csv_path: str = "datasets/tagged.csv",
        *,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """Train event type classifier.

        Default training data comes from:
        - `datasets/id_and_headline_first_sentence (1).csv` (id, headline)
        - `datasets/tagged.csv` (id; kategoria; ...)

        Data is split 80/20 (train/test) by default.
        """

        df = load_event_type_training_frame(
            headlines_csv_path=headlines_csv_path,
            tagged_csv_path=tagged_csv_path,
        )

        print(f"\n=== DANE TRENINGOWE (połączone) ===")
        print(f"Liczba przykładów: {len(df)}")
        print("Najczęstsze etykiety:")
        print(df["label"].value_counts().head(10))

        if not isinstance(self.classifier, TrainableEventTypeClassifier) and not hasattr(self.classifier, "train"):
            raise TypeError("Aktualny klasyfikator nie obsługuje trenowania (np. tryb Ollama).")

        # Typowanie: train jest dostępne dla sklearnowego klasyfikatora.
        self.classifier.train(  # type: ignore[attr-defined]
            df["sentence"].tolist(),
            df["label"].tolist(),
            test_size=test_size,
            random_state=random_state,
            stratify=True,
        )

    def extract_event(self, sentence: str) -> EventRecord:
        """Zwróć `EventRecord` dla zdania.

        Najpierw klasyfikujemy typ zdarzenia. Następnie zawsze próbujemy wyciągnąć relacje.
        Dla klasy `BRAK_ZDARZENIA` zachowujemy relacje czasu/miejsca (jeśli model je znajdzie),
        ale KTO/CO/TRIGGER ustawiamy na None.
        """
        event_type, confidence = self.classifier.predict(sentence)

        who, trigger, what, where, when = self.relations.extract_relations(sentence)

        if event_type == "BRAK_ZDARZENIA":
            # Gdy nie ma zdarzenia, WHO/WHAT/TRIGGER zwykle nie mają sensu.
            return EventRecord(
                event_type=event_type,
                who=None,
                what=None,
                trigger=None,
                where=where,
                when=when,
                confidence=confidence,
                sentence=sentence
            )

        return EventRecord(
            event_type=event_type,
            who=who,
            what=what,
            trigger=trigger,
            where=where,
            when=when,
            confidence=confidence,
            sentence=sentence
        )

    def save_classifier(self, path: str) -> None:
        """Zapisz wytrenowany klasyfikator typu zdarzenia."""
        if hasattr(self.classifier, "save"):
            self.classifier.save(path)  # type: ignore[attr-defined]
            return
        raise TypeError("Aktualny klasyfikator nie obsługuje zapisu (np. tryb Ollama).")

    def load_classifier(self, path: str) -> None:
        """Wczytaj zapisany klasyfikator typu zdarzenia."""
        if hasattr(self.classifier, "load"):
            self.classifier.load(path)  # type: ignore[attr-defined]
            return
        raise TypeError("Aktualny klasyfikator nie obsługuje wczytywania (np. tryb Ollama).")
