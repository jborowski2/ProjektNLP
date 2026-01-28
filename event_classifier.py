from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import TransformerMixin
from typing import List, Tuple, Optional, cast
from collections import Counter
import numpy as np
import joblib

from compat import ensure_supported_python

ensure_supported_python()

try:
    from spacy.lang.pl.stop_words import STOP_WORDS as POLISH_STOPWORDS
except Exception:
    POLISH_STOPWORDS = set()


class EventClassifier:
    def __init__(self):
        self.vectorizer = FeatureUnion(
            [
                (
                    "word",
                    cast(
                        TransformerMixin,
                        TfidfVectorizer(
                        stop_words=list(POLISH_STOPWORDS),
                        ngram_range=(1, 2),
                        lowercase=True,
                        min_df=2,
                        max_df=0.95,
                        ),
                    ),
                ),
                (
                    "char",
                    cast(
                        TransformerMixin,
                        TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        lowercase=True,
                        min_df=2,
                        max_df=0.95,
                        ),
                    ),
                ),
            ]
        )

        self.classifier = LogisticRegression(
            max_iter=8000,
            solver="saga",
            class_weight="balanced",
        )
        self.is_trained = False
        self.meta: dict[str, object] = {}

        # Light keyword overrides for very rare classes.
        # Applied only when the ML confidence is low.
        self._keyword_overrides: list[tuple[str, list[str]]] = [
            ("PRZESTEPSTWO", ["napastnik", "pobił", "pobity", "kradzie", "właman", "złodziej", "rabun", "napad"]),
            ("WYPADEK", ["wypadek", "zderzy", "uderzy", "potrąci", "kolizj", "samochód", "kierowca", "autobus", "ciężarówk"]),
            ("KATASTROFA", ["pożar", "zginę", "wybuch", "katastrof", "zawali", "powódź", "trzęsienie"]),
            ("PROTEST", ["protest", "strajk", "manifestac", "demonstrac"]),
        ]

    def save(self, path: str, *, meta: Optional[dict[str, object]] = None) -> None:
        payload = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "is_trained": self.is_trained,
            "keyword_overrides": self._keyword_overrides,
            "meta": meta or self.meta,
        }
        joblib.dump(payload, path)

    def load(self, path: str) -> None:
        payload = joblib.load(path)
        self.vectorizer = payload["vectorizer"]
        self.classifier = payload["classifier"]
        self.is_trained = bool(payload.get("is_trained", True))
        self._keyword_overrides = payload.get("keyword_overrides", self._keyword_overrides)
        self.meta = payload.get("meta", {})

    def train(
        self,
        sentences: List[str],
        labels: List[str],
        *,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        oversample: bool = True,
    ):
        stratify_labels = labels if stratify else None

        try:
            s_train, s_test, y_train, y_test = train_test_split(
                sentences,
                labels,
                test_size=test_size,
                stratify=stratify_labels,
                random_state=random_state,
            )
        except ValueError:
            s_train, s_test, y_train, y_test = train_test_split(
                sentences,
                labels,
                test_size=test_size,
                stratify=None,
                random_state=random_state,
            )

        X_train = self.vectorizer.fit_transform(s_train)
        X_test = self.vectorizer.transform(s_test)

        if oversample:
            X_train, y_train = self._oversample_train_set(
                X_train,
                y_train,
                random_state=random_state,
            )

        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.classifier.predict(X_test)

        print("\n=== KLASYFIKACJA ZDARZEŃ ===")
        print(classification_report(y_test, y_pred, zero_division=0))

    def _oversample_train_set(self, X_train, y_train, *, random_state: int):
        """Random oversampling to the max class count.

        Keeps the test set untouched. Works with scipy sparse matrices.
        """

        counts = Counter(y_train)
        if not counts:
            return X_train, y_train

        target = max(counts.values())
        if target <= 1:
            return X_train, y_train

        rng = np.random.default_rng(random_state)

        indices_by_class: dict[str, np.ndarray] = {}
        y_arr = np.asarray(y_train)
        for label in counts.keys():
            indices_by_class[label] = np.flatnonzero(y_arr == label)

        extra_indices: list[np.ndarray] = []
        for label, idxs in indices_by_class.items():
            if len(idxs) == 0:
                continue
            missing = target - len(idxs)
            if missing <= 0:
                continue
            sampled = rng.choice(idxs, size=missing, replace=True)
            extra_indices.append(sampled)

        if not extra_indices:
            return X_train, y_train

        all_indices = np.concatenate([np.arange(len(y_train)), *extra_indices])
        rng.shuffle(all_indices)

        X_res = X_train[all_indices]
        y_res = y_arr[all_indices].tolist()

        print("\n=== BALANSOWANIE KLAS (oversampling) ===")
        print("Przed:")
        print(Counter(y_train))
        print("Po:")
        print(Counter(y_res))

        return X_res, y_res

    def predict(self, sentence: str) -> Tuple[str, float]:
        if not self.is_trained:
            raise RuntimeError("Classifier is not trained. Train it first or load a saved model.")
        vec = self.vectorizer.transform([sentence])

        if hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(vec)[0]
            label = self.classifier.classes_[int(np.argmax(proba))]
            prob = float(np.max(proba))
        elif hasattr(self.classifier, "decision_function"):
            scores = self.classifier.decision_function(vec)
            scores = np.asarray(scores).reshape(-1)
            # softmax -> pseudo-probabilities
            scores = scores - np.max(scores)
            exp = np.exp(scores)
            proba = exp / np.sum(exp)
            label = self.classifier.classes_[int(np.argmax(proba))]
            prob = float(np.max(proba))
        else:
            label = self.classifier.predict(vec)[0]
            prob = 1.0

        overridden = self._maybe_override_with_keywords(sentence, current_label=label, current_prob=prob)
        if overridden is not None:
            return overridden

        return str(label), prob

    def _maybe_override_with_keywords(
        self,
        sentence: str,
        *,
        current_label: str,
        current_prob: float,
        threshold: float = 0.55,
    ) -> Optional[Tuple[str, float]]:
        if current_prob >= threshold:
            return None

        s = sentence.lower()
        for label, keywords in self._keyword_overrides:
            if any(k in s for k in keywords):
                # Only override if it would actually change the class.
                if label != current_label:
                    return label, 0.90
        return None
