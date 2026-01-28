from __future__ import annotations

"""Proste GUI (PySide6) do analizy pojedynczych zdań.

Założenia UX:
- użytkownik wybiera model z listy; model jest wczytywany automatycznie,
- po lewej: pole tekstowe + przycisk analizy,
- po prawej: szczegóły modelu i metryki (z plików leaderboard).

GUI nie trenuje modeli — zakłada, że w `models/` istnieją zapisane `.joblib`.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QComboBox,
    QGroupBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import pandas as pd

from event_extractor import EventExtractor


DEFAULT_MODEL_PATH = Path("models/event_type_model.joblib")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_LEADERBOARD_PATHS = [
    Path("results/leaderboard.csv"),
    Path("results/leaderboard_both.csv"),
    Path("results/leaderboard_full.csv"),
    Path("results/experiments_summary.csv"),
    Path("results/experiments_both_summary.csv"),
    Path("results/experiments_full_summary.csv"),
]


def _safe_read_csv(path: Path, *, sep: str) -> pd.DataFrame | None:
    """Bezpiecznie wczytaj CSV (GUI nie powinno się wywracać od braku pliku)."""
    try:
        if not path.exists():
            return None
        return pd.read_csv(path, sep=sep)
    except Exception:
        return None


def _model_key_from_path(p: Path) -> str:
    """Zamień ścieżkę pliku modelu na klucz modelu (bez prefixu i timestampu).

    Przykład:
    - models/all_LogReg_L2_20260128_005432.joblib -> LogReg_L2
    """
    name = p.stem
    if name.startswith("all_"):
        name = name[len("all_") :]
    # Drop timestamp suffix like _20260127_203947
    parts = name.split("_")
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit() and len(parts[-2]) == 8:
        name = "_".join(parts[:-2])
    return name


def _friendly_model_name(model_key: str) -> str:
    """Czytelna nazwa modelu do wyświetlenia w UI.

    Examples:
      - lr_wordchar_balanced_over -> Regresja logistyczna (word+char, balanced, oversampling)
      - svm_word_balanced_over -> SVM liniowy (word, balanced, oversampling)
      - event_type_model -> Model domyślny
    """

    key = (model_key or "").strip()
    if not key:
        return "(nieznany model)"

    if key == "event_type_model":
        return "Model domyślny"

    # Keys coming from old naming scheme OR from experiments cfg.name values.
    exact_map = {
        "LinearSVC": "SVM liniowy",
        "Tuned_LinearSVC": "SVM liniowy (tuned)",
        "LogReg_L2": "Regresja logistyczna (L2)",
        "Tuned_LogReg_L2": "Regresja logistyczna (L2, tuned)",
        "MultinomialNB": "Naiwny Bayes (MultinomialNB)",
        "Bagging_LogReg": "Bagging (Regresja logistyczna)",
        "GradientBoosting": "Gradient Boosting",
        "MLP": "Sieć neuronowa (MLP)",
        "XGBoost": "XGBoost",
        "Tuned_XGBoost": "XGBoost (tuned)",
        "lr_word_balanced": "Regresja logistyczna (word, balanced)",
        "lr_word_balanced_over": "Regresja logistyczna (word, balanced, oversampling)",
        "lr_wordchar_balanced_over": "Regresja logistyczna (word+char, balanced, oversampling)",
        "svm_word_balanced_over": "SVM liniowy (word, balanced, oversampling)",
        "svm_wordchar_balanced_over": "SVM liniowy (word+char, balanced, oversampling)",
        "mnb_word_over": "Naiwny Bayes (word, oversampling)",
    }
    if key in exact_map:
        return exact_map[key]

    base_map = {
        "lr": "Regresja logistyczna",
        "svm": "SVM liniowy",
        "mnb": "Naiwny Bayes (MultinomialNB)",
        "xgboost": "XGBoost",
        "linearsvc": "SVM liniowy",
        "logreg": "Regresja logistyczna",
        "gradientboosting": "Gradient Boosting",
        "mlp": "Sieć neuronowa (MLP)",
        "bagging": "Bagging",
    }

    parts = key.split("_")
    base = parts[0]
    base_label = base_map.get(base, key)

    # Vectorizer
    vec_label = None
    if "wordchar" in key or "word_char" in key:
        vec_label = "word+char"
    elif "word" in key:
        vec_label = "word"

    flags: list[str] = []
    if "balanced" in key:
        flags.append("balanced")
    if "over" in key or "oversample" in key:
        flags.append("oversampling")
    if "svd" in key:
        flags.append("svd")

    details: list[str] = []
    if vec_label:
        details.append(vec_label)
    details.extend(flags)

    if details and base_label != key:
        return f"{base_label} ({', '.join(details)})"
    return base_label


def _fmt_float(x: Any) -> str:
    try:
        val = float(x)
        if val != val:  # nan
            return ""
        return f"{val:.3f}"
    except Exception:
        return ""


def _fmt_dt(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def _fmt_size(n: int) -> str:
    try:
        n = int(n)
    except Exception:
        return ""
    units = ["B", "KB", "MB", "GB"]
    size = float(n)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:.1f} {u}" if u != "B" else f"{int(size)} {u}"
        size /= 1024
    return f"{n} B"


def _load_leaderboard_rows() -> dict[str, dict[str, Any]]:
    """Wczytaj metryki z plików leaderboard.

    Zwraca mapę: model_key -> słownik z kolumnami (Accuracy, F1_*, ...).
    Jeśli jest kilka plików, późniejszy może nadpisać wcześniejszy wpis tego samego modelu.
    """
    rows: dict[str, dict[str, Any]] = {}
    for p in DEFAULT_LEADERBOARD_PATHS:
        df = _safe_read_csv(p, sep=",")
        if df is None or df.empty:
            continue
        # Expected columns: Model,n,Accuracy,F1_weighted,F1_macro,AIC,BIC
        if "Model" not in df.columns:
            continue
        for _, r in df.iterrows():
            key = str(r.get("Model", "")).strip()
            if not key:
                continue
            rows[key] = {k: r.get(k) for k in df.columns}
            rows[key]["__source"] = str(p)
    return rows


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ProjektNLP – Klasyfikacja zdarzeń")

        self.extractor = EventExtractor()
        self.model_path = DEFAULT_MODEL_PATH
        self.models_dir = DEFAULT_MODELS_DIR
        self._leaderboard = _load_leaderboard_rows()
        self._model_paths: list[Path] = []

        root = QWidget()
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)

        header = QLabel("ProjektNLP – klasyfikacja zdarzeń + ekstrakcja relacji")
        header.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        # LEFT: controls + input + output
        left = QWidget()
        left_layout = QVBoxLayout(left)
        splitter.addWidget(left)

        model_box = QGroupBox("Model")
        left_layout.addWidget(model_box)
        mb = QVBoxLayout(model_box)

        top_row = QHBoxLayout()
        self.model_count = QLabel("Modele: 0")
        self.model_count.setStyleSheet("color: #94a3b8;")
        top_row.addWidget(self.model_count)
        top_row.addStretch(1)
        mb.addLayout(top_row)

        mb.addWidget(QLabel("Wybierz model:"))
        row0 = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        row0.addWidget(self.model_combo, 1)
        mb.addLayout(row0)

        input_box = QGroupBox("Tekst")
        left_layout.addWidget(input_box)
        ib = QVBoxLayout(input_box)
        ib.addWidget(QLabel("Wpisz zdanie:"))
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Np. Napastnik pobił ochroniarza przed klubem.")
        self.input_text.setFixedHeight(110)
        ib.addWidget(self.input_text)

        self.btn_analyze = QPushButton("Analizuj (typ + KTO/CO/GDZIE/KIEDY)")
        self.btn_analyze.clicked.connect(self.on_analyze)
        ib.addWidget(self.btn_analyze)

        out_box = QGroupBox("Wynik")
        left_layout.addWidget(out_box, 1)
        ob = QVBoxLayout(out_box)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(220)
        self.output.setPlaceholderText("Tu pojawi się wynik…")
        ob.addWidget(self.output)

        # RIGHT: model stats
        right = QWidget()
        right_layout = QVBoxLayout(right)
        splitter.addWidget(right)
        stats_box = QGroupBox("Statystyki modelu")
        right_layout.addWidget(stats_box, 1)
        sb = QVBoxLayout(stats_box)
        self.model_stats = QTextEdit()
        self.model_stats.setReadOnly(True)
        self.model_stats.setPlaceholderText("Wybierz model, żeby zobaczyć statystyki…")
        sb.addWidget(self.model_stats)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.statusBar().showMessage("Gotowe")

        self._apply_styles()

        self.refresh_models()
        self.on_model_changed()

    def _apply_styles(self) -> None:
        # Lightweight styling to make the UI feel less default.
        self.setStyleSheet(
            """
            QMainWindow { background: #0f172a; }
            QLabel { color: #e2e8f0; }
            QGroupBox { color: #e2e8f0; border: 1px solid #334155; border-radius: 8px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px 0 6px; }
            QTextEdit, QLineEdit, QComboBox {
                background: #0b1220; color: #e2e8f0; border: 1px solid #334155; border-radius: 8px;
                padding: 8px;
            }
            QPushButton {
                background: #1d4ed8; color: white; border: 0px; border-radius: 8px; padding: 8px 12px;
            }
            QPushButton:hover { background: #2563eb; }
            QPushButton:disabled { background: #334155; color: #94a3b8; }
            QStatusBar { color: #e2e8f0; }
            """
        )

    def _model_label_text(self) -> str:
        return f"Model: {self.model_path}"

    def refresh_models(self) -> None:
        """Zbuduj listę modeli z katalogu models/.

        Uwaga: w comboboxie pokazujemy tylko nazwy przyjazne; szczegóły są po prawej.
        """
        self.model_combo.clear()
        if not self.models_dir.exists():
            self.model_combo.addItem("(brak katalogu models/)")
            self.model_count.setText("Modele: 0")
            return

        paths = sorted(self.models_dir.glob("**/*.joblib"))
        if not paths:
            self.model_combo.addItem("(brak zapisanych modeli .joblib)")
            self.model_count.setText("Modele: 0")
            return

        self._model_paths = paths
        shown = 0

        # Keep combo labels clean (no metrics, no filenames); disambiguate duplicates.
        name_counts: dict[str, int] = {}
        for p in paths:
            key = _model_key_from_path(p)
            friendly = _friendly_model_name(key)

            idx = name_counts.get(friendly, 0) + 1
            name_counts[friendly] = idx

            display = friendly if idx == 1 else f"{friendly} ({idx})"
            self.model_combo.addItem(display, str(p))
            shown += 1

        self.model_count.setText(f"Modele: {shown}")

        # Prefer the default model if present
        try:
            default_path = DEFAULT_MODEL_PATH.resolve()
        except Exception:
            default_path = DEFAULT_MODEL_PATH

        for i in range(self.model_combo.count()):
            data = self.model_combo.itemData(i)
            if not data:
                continue
            try:
                item_path = Path(str(data)).resolve()
            except Exception:
                item_path = Path(str(data))
            if item_path == default_path:
                self.model_combo.setCurrentIndex(i)
                break

    def on_model_changed(self) -> None:
        """Obsługa zmiany wyboru w comboboxie.

        - ustawia `self.model_path`,
        - ładuje klasyfikator do `EventExtractor`,
        - odświeża panel statystyk.
        """
        data = self.model_combo.currentData() if hasattr(self, "model_combo") else None
        if not data:
            self.model_stats.setPlainText("(brak wybranego modelu)")
            return

        selected = Path(str(data))
        if not selected.exists():
            self.model_stats.setPlainText(f"(nie znaleziono pliku modelu: {selected})")
            return

        self.model_path = selected
        try:
            self.extractor.load_classifier(str(self.model_path))
            self.statusBar().showMessage("Model wczytany")
        except Exception as exc:
            QMessageBox.critical(self, "Błąd", f"Nie udało się wczytać modelu: {exc}")

        self._update_model_stats_panel()

    def _ensure_model_loaded(self) -> bool:
        try:
            # A cheap check: predict() will raise if not trained
            if not self.extractor.classifier.is_trained:
                raise RuntimeError("Model nie jest wczytany")
            return True
        except Exception:
            return False

    def _update_model_stats_panel(self) -> None:
        data = self.model_combo.currentData() if hasattr(self, "model_combo") else None
        if not data:
            self.model_stats.setPlainText("(brak wybranego modelu)")
            return
        p = Path(str(data))
        key = _model_key_from_path(p)

        lines: list[str] = []
        lines.append(f"Nazwa: {key}")
        lines.append(f"Plik: {p.as_posix()}")
        try:
            st = p.stat()
            lines.append(f"Rozmiar: {_fmt_size(st.st_size)}")
            lines.append(f"Modyfikacja: {_fmt_dt(st.st_mtime)}")
        except Exception:
            pass

        stats = self._leaderboard.get(key)
        if stats:
            src = stats.get("__source", "")
            lines.append("")
            lines.append("Wyniki (z leaderboard):")
            for k in ["Accuracy", "F1_weighted", "F1_macro", "AIC", "BIC", "n"]:
                if k in stats:
                    v = stats.get(k)
                    if k in {"Accuracy", "F1_weighted", "F1_macro"}:
                        lines.append(f"- {k}: {_fmt_float(v)}")
                    else:
                        lines.append(f"- {k}: {v}")
            if src:
                lines.append(f"Źródło: {src}")
        else:
            lines.append("")
            lines.append("Wyniki: (brak wpisu w results/leaderboard*.csv dla tego modelu)")

        # Optional: peek into joblib for basic info (safe / best-effort).
        try:
            import joblib

            payload = joblib.load(p)
            clf = payload.get("classifier") if isinstance(payload, dict) else None
            vec = payload.get("vectorizer") if isinstance(payload, dict) else None
            meta = payload.get("meta") if isinstance(payload, dict) else None
            lines.append("")
            lines.append("Szczegóły (z pliku .joblib):")
            if clf is not None:
                lines.append(f"- Klasyfikator: {type(clf).__name__}")
            if vec is not None:
                lines.append(f"- Wektoryzator: {type(vec).__name__}")
            if isinstance(meta, dict) and meta:
                lines.append("- Meta:")
                for mk, mv in list(meta.items())[:12]:
                    lines.append(f"  • {mk}: {mv}")
        except Exception:
            # Don't spam errors; leaderboard usually enough.
            pass

        self.model_stats.setPlainText("\n".join(lines))

    def _get_sentence(self) -> str:
        return self.input_text.toPlainText().strip()

    def on_classify(self) -> None:
        sentence = self._get_sentence()
        if not sentence:
            return

        if not self._ensure_model_loaded():
            QMessageBox.information(
                self,
                "Model nie wczytany",
                "Wybierz model z listy.",
            )
            return

        try:
            label, prob = self.extractor.classifier.predict(sentence)
            self.output.setPlainText(f"Typ zdarzenia: {label}\nPewność: {prob:.2f}\n\nZdanie:\n{sentence}")
        except Exception as exc:
            QMessageBox.critical(self, "Błąd", f"Klasyfikacja nie powiodła się: {exc}")

    def on_extract(self) -> None:
        sentence = self._get_sentence()
        if not sentence:
            return

        if not self._ensure_model_loaded():
            QMessageBox.information(
                self,
                "Model nie wczytany",
                "Wybierz model z listy.",
            )
            return

        try:
            event = self.extractor.extract_event(sentence)
            self.output.setPlainText(str(event))
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Błąd",
                "Ekstrakcja nie powiodła się. Upewnij się, że masz pobrany model spaCy 'pl_core_news_lg'.\n\n"
                f"Szczegóły: {exc}",
            )

    def on_analyze(self) -> None:
        sentence = self._get_sentence()
        if not sentence:
            return

        if not self._ensure_model_loaded():
            QMessageBox.information(
                self,
                "Model nie wczytany",
                "Wybierz model z listy.",
            )
            return

        try:
            event = self.extractor.extract_event(sentence)
            self.output.setPlainText(str(event))
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Błąd",
                "Analiza nie powiodła się. Upewnij się, że masz pobrany model spaCy 'pl_core_news_lg'.\n\n"
                f"Szczegóły: {exc}",
            )


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(980, 640)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
