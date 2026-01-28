from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional


class OllamaError(RuntimeError):
    pass


@dataclass(frozen=True)
class OllamaClient:
    """Minimalny klient lokalnej Ollamy przez HTTP.

    Domyślnie Ollama nasłuchuje na `http://localhost:11434`.
    Możesz nadpisać przez zmienną środowiskową `OLLAMA_HOST`.
    """

    host: str = ""
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if not self.host:
            object.__setattr__(self, "host", os.environ.get("OLLAMA_HOST", "http://localhost:11434"))

    def chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        format: Optional[str] = "json",
        options: Optional[dict[str, Any]] = None,
    ) -> str:
        """Wywołaj `/api/chat` i zwróć tekst odpowiedzi modelu."""

        payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if format is not None:
            payload["format"] = format
        if options:
            payload["options"] = options

        url = self.host.rstrip("/") + "/api/chat"
        body = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            raise OllamaError(
                "Nie mogę połączyć się z Ollamą. "
                "Upewnij się, że Ollama działa (Windows: aplikacja/serwis), "
                "a host jest poprawny (domyślnie http://localhost:11434). "
                f"Szczegóły: {exc}"
            ) from exc

        try:
            data = json.loads(raw)
        except Exception as exc:
            raise OllamaError(f"Ollama zwróciła nie-JSON: {raw[:500]}") from exc

        msg = (data.get("message") or {}).get("content")
        if not isinstance(msg, str) or not msg.strip():
            raise OllamaError(f"Nieoczekiwana odpowiedź Ollamy: {data}")
        return msg.strip()
