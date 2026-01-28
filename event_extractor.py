import pandas as pd
from typing import Optional
from event_record import EventRecord
from event_classifier import EventClassifier
from relation_extractor import RelationExtractor
from data_loading import load_event_type_training_frame


class EventExtractor:
    def __init__(self):
        self.classifier = EventClassifier()
        self.relations = RelationExtractor()

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

        self.classifier.train(
            df["sentence"].tolist(),
            df["label"].tolist(),
            test_size=test_size,
            random_state=random_state,
            stratify=True,
        )

    def extract_event(self, sentence: str) -> EventRecord:
        event_type, confidence = self.classifier.predict(sentence)

        who, trigger, what, where, when = self.relations.extract_relations(sentence)

        if event_type == "BRAK_ZDARZENIA":
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
        self.classifier.save(path)

    def load_classifier(self, path: str) -> None:
        self.classifier.load(path)
