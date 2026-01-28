import sys


def ensure_supported_python() -> None:
    """Fail fast on interpreter versions known to break core deps.

    The project relies on spaCy, which (as of early 2026) depends on Pydantic v1
    compatibility that isn't working on Python 3.14+.
    """

    if sys.version_info >= (3, 14):
        raise SystemExit(
            "Python 3.14+ is currently not supported by this project (spaCy/Pydantic v1 compatibility). "
            "Install Python 3.12 or 3.13, recreate the venv, then reinstall requirements."
        )
