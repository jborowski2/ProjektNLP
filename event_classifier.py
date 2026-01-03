from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import List, Tuple
import numpy as np


class EventClassifier:
    def __init__(self, model_name: str):
        self.encoder = SentenceTransformer(model_name)
        self.classifier = LogisticRegression(max_iter=1000)
        self.is_trained = False

    def train(self, sentences: List[str], labels: List[str]):
        X = self.encoder.encode(sentences)

        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, stratify=labels, random_state=42
        )

        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.classifier.predict(X_test)

        print("\n=== KLASYFIKACJA ZDARZEŃ ===")
        print(
            classification_report(
                y_test,
                y_pred,
                zero_division=0
            )
        )

    def predict(self, sentence: str) -> Tuple[str, float]:
        vec = self.encoder.encode([sentence])
        label = self.classifier.predict(vec)[0]
        prob = max(self.classifier.predict_proba(vec)[0])
        return label, prob
