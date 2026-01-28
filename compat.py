"""Kompatybilność środowiska.

Ten moduł zawiera małe „bezpieczniki”, które wykrywają niekompatybilne wersje
interpretera Pythona zanim projekt uruchomi ciężkie zależności (spaCy / sklearn).
"""

import sys


def ensure_supported_python() -> None:
    """Sprawdź wersję Pythona i przerwij działanie, jeśli jest nieobsługiwana.

    Projekt korzysta m.in. ze spaCy, które (stan na 2026) ma problemy
    kompatybilności na Pythonie 3.14+ (zależności wokół Pydantic v1).
    """

    if sys.version_info >= (3, 14):
        raise SystemExit(
            "Python 3.14+ is currently not supported by this project (spaCy/Pydantic v1 compatibility). "
            "Install Python 3.12 or 3.13, recreate the venv, then reinstall requirements."
        )
