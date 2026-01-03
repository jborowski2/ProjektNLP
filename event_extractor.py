import pandas as pd
from typing import Optional
from event_record import EventRecord
from event_classifier import EventClassifier
from relation_extractor import RelationExtractor


class EventExtractor:
    def __init__(self):
        self.classifier = EventClassifier(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.relations = RelationExtractor()

    def train(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df = df.drop_duplicates()
        self.classifier.train(df["sentence"].tolist(), df["label"].tolist())

    def extract_event(self, sentence: str) -> EventRecord:
        event_type, confidence = self.classifier.predict(sentence)

        if event_type == "BRAK_ZDARZENIA":
            return EventRecord(
                event_type=event_type,
                who=None,
                what=None,
                trigger=None,
                confidence=confidence,
                sentence=sentence
            )

        who, trigger, what, where, when = self.relations.extract_relations(sentence)

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
