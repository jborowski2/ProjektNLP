from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple
import spacy
from spacy.lang.pl.stop_words import STOP_WORDS as POLISH_STOPWORDS


class EventClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words=list(POLISH_STOPWORDS),
            ngram_range=(1, 2)
        )

        self.classifier = LogisticRegression(max_iter=1000)
        self.is_trained = False

    def train(self, sentences: List[str], labels: List[str]):
        X = self.vectorizer.fit_transform(sentences)

        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, stratify=labels, random_state=42
        )

        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.classifier.predict(X_test)

        print("\n=== KLASYFIKACJA ZDARZEŃ ===")
        print(classification_report(y_test, y_pred, zero_division=0))

    def predict(self, sentence: str) -> Tuple[str, float]:
        vec = self.vectorizer.transform([sentence])
        label = self.classifier.predict(vec)[0]
        prob = max(self.classifier.predict_proba(vec)[0])
        return label, prob
