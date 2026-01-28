from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from ollama_client import OllamaClient, OllamaError


def _safe_json_loads(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def _none_if_empty(v: object) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"null", "none"}:
        return None
    return s


@dataclass
class OllamaRelationExtractor:
    """Ekstrakcja KTO/CO/TRIGGER/GDZIE/KIEDY z użyciem lokalnego LLM (Ollama)."""

    model: str = "qwen2.5:7b-instruct"
    host: str = ""
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        self.client = OllamaClient(host=self.host, timeout_s=self.timeout_s)

    def extract_relations(
        self, sentence: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        system = (
            "Jesteś ekstraktorem informacji ze zdania po polsku. "
            "Wyciągnij pola zdarzenia: KTO (sprawca/podmiot), CO (obiekt/dopełnienie), "
            "TRIGGER (czasownik/akcja), GDZIE (lokalizacja), KIEDY (czas). "
            "Zwróć WYŁĄCZNIE poprawny JSON bez dodatkowego tekstu. "
            "Nie zmyślaj: jeśli czegoś nie ma w zdaniu, ustaw null. "
            "TRIGGER zwróć najlepiej jako bezokolicznik (np. 'uderzyć', 'pobić')."
        )
        user = (
            "Zwróć JSON w formacie: "
            "{\"who\": string|null, \"what\": string|null, \"trigger\": string|null, "
            "\"where\": string|null, \"when\": string|null}.\n\n"
            f"Zdanie: {sentence}"
        )

        raw = self.client.chat(
            model=self.model,
            system=system,
            user=user,
            format="json",
            options={
                "temperature": 0.0,
                "top_p": 0.9,
                "num_predict": 192,
            },
        )

        try:
            obj = _safe_json_loads(raw)
        except Exception:
            return None, None, None, None, None

        who = _none_if_empty(obj.get("who"))
        what = _none_if_empty(obj.get("what"))
        trigger = _none_if_empty(obj.get("trigger"))
        where = _none_if_empty(obj.get("where"))
        when = _none_if_empty(obj.get("when"))

        return who, trigger, what, where, when
