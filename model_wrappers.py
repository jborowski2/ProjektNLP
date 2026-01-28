from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder


@dataclass
class LabelEncodedClassifier:
    """Wrap a classifier to train on encoded integer labels but predict original labels.

    Useful for estimators that are picky about y dtype (some XGBoost builds).
    """

    estimator: Any
    label_encoder: LabelEncoder = LabelEncoder()

    def fit(self, X, y):
        y_enc = self.label_encoder.fit_transform([str(v) for v in y])
        self.estimator.fit(X, y_enc)
        self.classes_ = np.array(self.label_encoder.classes_, dtype=object)
        return self

    def predict(self, X):
        y_pred = self.estimator.predict(X)
        y_pred = np.asarray(y_pred)
        return self.label_encoder.inverse_transform(y_pred.astype(int))

    def predict_proba(self, X):
        proba = self.estimator.predict_proba(X)
        return proba

    def decision_function(self, X):
        if hasattr(self.estimator, "decision_function"):
            return self.estimator.decision_function(X)
        raise AttributeError("Underlying estimator has no decision_function")

    def get_booster(self):
        if hasattr(self.estimator, "get_booster"):
            return self.estimator.get_booster()
        raise AttributeError("Underlying estimator has no get_booster")
