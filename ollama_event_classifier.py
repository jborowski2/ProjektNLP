from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

from data_loading import load_event_type_labels
from ollama_client import OllamaClient, OllamaError


def _clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _safe_json_loads(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        # Często model „owija” JSON w dodatkowy tekst. Spróbuj wyciąć pierwszą klamrę.
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


@dataclass
class OllamaEventClassifier:
    """Klasyfikator typu zdarzenia oparty o lokalny LLM (Ollama).

    API zgodne z EventClassifier: `predict(sentence) -> (label, confidence)`.
    """

    model: str = "qwen2.5:7b-instruct"
    labels: Sequence[str] | None = None
    host: str = ""
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if self.labels is None:
            self.labels = load_event_type_labels()
        self._label_set = {str(l) for l in self.labels}
        if not self._label_set:
            raise ValueError("Brak etykiet klas. Sprawdź datasets/tagged.csv (kolumna 'kategoria').")

        self.client = OllamaClient(host=self.host, timeout_s=self.timeout_s)

        # Preferowany fallback, jeśli jest w słowniku.
        self._fallback_label = "BRAK_ZDARZENIA" if "BRAK_ZDARZENIA" in self._label_set else sorted(self._label_set)[0]

    def predict(self, sentence: str) -> tuple[str, float]:
        allowed = ", ".join(sorted(self._label_set))

        system = (
            "Jesteś klasyfikatorem typu zdarzenia dla polskich nagłówków/zdania. "
            "Masz zwrócić WYŁĄCZNIE poprawny JSON, bez żadnych komentarzy ani dodatkowego tekstu. "
            "Pole 'label' musi być jedną z dozwolonych etykiet. "
            "Pole 'confidence' to liczba 0-1 (twoja subiektywna pewność)."
        )
        user = (
            f"Dozwolone etykiety: [{allowed}].\n"
            "Sklasyfikuj zdanie do jednej etykiety. Jeśli nie pasuje do żadnej, wybierz BRAK_ZDARZENIA (jeśli dostępne).\n"
            "Zwróć JSON w formacie: {\"label\": "
            "\"ETK\", \"confidence\": 0.0}.\n\n"
            f"Zdanie: {sentence}"
        )

        try:
            raw = self.client.chat(
                model=self.model,
                system=system,
                user=user,
                format="json",
                options={
                    "temperature": 0.0,
                    "top_p": 0.9,
                    # ogranicz gadatliwość
                    "num_predict": 128,
                },
            )
        except OllamaError:
            raise

        try:
            obj = _safe_json_loads(raw)
        except Exception:
            return self._fallback_label, 0.0

        label = str(obj.get("label", "")).strip()
        if label not in self._label_set:
            # Spróbuj normalizacji (często model doda spacje / zmieni wielkość liter).
            label_up = label.upper().strip()
            if label_up in self._label_set:
                label = label_up
            else:
                label = self._fallback_label

        conf = obj.get("confidence", 0.0)
        try:
            conf_f = _clamp01(float(conf))
        except Exception:
            conf_f = 0.0

        return label, conf_f
