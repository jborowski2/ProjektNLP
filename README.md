# ProjektNLP - Ekstrakcja wydarzeń z newsów

System do automatycznej ekstrakcji strukturalnych wydarzeń z polskich newsów (KTO, CO, GDZIE, KIEDY).

## Opis projektu

Projekt implementuje system do budowy rekordów wydarzeń z artykułów prasowych w języku polskim, zawierających:
- **Typ zdarzenia** (np. PRZESTĘPSTWO, WYPADEK, POLITYKA)
- **Sprawca (KTO)** - podmiot zdarzenia
- **Obiekt (CO)** - dopełnienie, na kim/czym wykonano akcję  
- **Trigger** - czasownik/akcja wywołująca zdarzenie
- **Opcjonalnie**: lokalizacja (GDZIE), czas (KIEDY)

### Przykład

**Zdanie wejściowe:**
```
Napastnik pobił ochroniarza przed klubem.
```

**Wyodrębnione wydarzenie:**
```
Typ zdarzenia: PRZESTĘPSTWO
KTO: napastnik
CO: ochroniarza
Trigger: pobił
```

## Architektura

System składa się z dwóch głównych komponentów:

### 1. Klasyfikacja typu wydarzenia
- Wykorzystuje cechy TF-IDF (unigramy + bigramy)
- Klasyfikator: Logistic Regression
- **Zbiór uczący: 220+ przykładów** (gotowy do rozszerzenia do 1000+)

### 2. Wykrywanie relacji (podmiot-orzeczenie-dopełnienie)
- Wykorzystuje Universal Dependencies parsing
- Model spaCy: `pl_core_news_lg`
- Reguły:
  - **KTO (X)**: `nsubj(V, X)` - nominal subject
  - **CO**: `obj(V, X)` lub `obl(V, X)` - direct object lub oblique nominal
- **Zbiór testowy: 105+ oznaczeń**

## Instalacja

### Wymagania
- Python 3.10–3.13 (zalecane: 3.12)
- pip

**Uwaga (Python 3.14 / Windows):** projekt opiera się o spaCy, które na Pythonie 3.14 potrafi obecnie wysypywać się przez kompatybilność z Pydantic v1. Najprościej: użyj Pythona 3.12 (albo 3.13).

### Kroki instalacji

1. Sklonuj repozytorium:
```bash
git clone https://github.com/jborowski2/ProjektNLP.git
cd ProjektNLP
```

2. Zainstaluj wymagane pakiety:
```bash
pip install -r requirements.txt
```

Jeśli korzystasz z VS Code, najpierw wybierz interpreter Pythona 3.12/3.13: `Python: Select Interpreter`, a dopiero potem utwórz venv (np. `python -m venv .venv`).

3. Pobierz model spaCy dla języka polskiego:
```bash
python setup_models.py
```

Alternatywnie można pobrać model ręcznie:
```bash
python -m spacy download pl_core_news_lg
```

## Troubleshooting

### Problem: błędy instalacji / importu na Python 3.14

Najczęstsza przyczyna na Windows to użycie Pythona 3.14 — spaCy (i zależności wokół Pydantic v1) mogą nie być jeszcze kompatybilne i import kończy się błędem.

**Rozwiązanie (zalecane):**
1. Zainstaluj Python 3.12 x64.
2. W VS Code wybierz interpreter Pythona 3.12: `Python: Select Interpreter`.
3. Usuń stare środowisko: skasuj folder `.venv`.
4. Utwórz venv i zainstaluj zależności ponownie:
  - `python -m venv .venv`
  - `./.venv/Scripts/python -m pip install -U pip`
  - `./.venv/Scripts/python -m pip install -r requirements.txt`

**Alternatywa (niezalecane):** próba „przepchania” Pythona 3.14 przez ręczne pinowanie zależności / kompilację — zwykle kończy się walką z kompatybilnością.

## Użycie

### Podstawowe użycie

```python
from event_extractor import EventExtractor

# Inicjalizacja
extractor = EventExtractor()

# Trenowanie klasyfikatora
extractor.train_classifier("datasets/training_data.csv")

# Ekstrakcja wydarzenia z pojedynczego zdania
sentence = "Napastnik pobił ochroniarza przed klubem."
event = extractor.extract_event(sentence)
print(event)
```

### Uruchomienie pełnego przykładu

```bash
python event_extractor.py
```

## Eksperymenty i wybór podejścia

Projekt zakłada wykonywanie eksperymentów (różne modele/cechy/parametry) i wybór najlepszego podejścia na podstawie miar jakości.

- Protokół i opis miar: [EXPERIMENTS.md](EXPERIMENTS.md)
- Uruchomienie: `./.venv/Scripts/python experiments.py`
- Wyniki: `results/experiments.csv`

## Aplikacja okienkowa (Python + Qt)

Aplikacja GUI ładuje wytrenowany model i pozwala klasyfikować zdania oraz (opcjonalnie) wyciągać KTO/CO/TRIGGER/GDZIE/KIEDY.

1. Wytrenuj i zapisz model:
  - `./.venv/Scripts/python train_and_save_model.py --out models/event_type_model.joblib`
2. Uruchom GUI:
  - `./.venv/Scripts/python qt_app.py`
   
Model można:
- wczytać z pliku (przycisk "Wczytaj model…"), albo
- wybrać z listy modeli zapisanych w katalogu `models/` (lista rozwijana + "Wczytaj wybrany").

Żeby mieć kilka wytrenowanych modeli do wyboru, możesz je zapisać z `experiments.py`:

`./.venv/Scripts/python experiments.py --save-all-models`

albo zapisać tylko najlepszy wg macro-F1:

`./.venv/Scripts/python experiments.py --save-best-model`

Jeśli chcesz używać ekstrakcji relacji, upewnij się, że masz pobrany model spaCy:
`./.venv/Scripts/python -m spacy download pl_core_news_lg`

Ten skrypt:
1. Trenuje klasyfikator na zbiorze treningowym
2. Ewaluuje ekstrakcję relacji na zbiorze testowym
3. Pokazuje przykłady ekstrakcji wydarzeń
4. Zapisuje wytrenowany model

### API poszczególnych komponentów

#### Ekstrakcja relacji (Universal Dependencies)

```python
from relation_extractor import RelationExtractor

extractor = RelationExtractor()
who, trigger, what = extractor.extract_who_what(
    "Policjant zatrzymał złodzieja na ulicy."
)
print(f"KTO: {who}, Trigger: {trigger}, CO: {what}")
```

#### Klasyfikacja typu wydarzenia

```python
from event_classifier import EventClassifier
import pandas as pd

classifier = EventClassifier()

# Trenowanie
df = pd.read_csv("datasets/training_data.csv")
classifier.train(df['sentence'].tolist(), df['label'].tolist())

# Predykcja
event_type, confidence = classifier.predict("Samochód uderzył w drzewo.")
print(f"Typ: {event_type}, Pewność: {confidence:.2f}")
```

## Struktura projektu

```
ProjektNLP/
├── README.md                      # Ten plik
├── requirements.txt               # Zależności Python
├── setup_models.py               # Skrypt do pobierania modeli
├── event_record.py               # Model danych EventRecord
├── relation_extractor.py         # Ekstrakcja relacji (UD)
├── event_classifier.py           # Klasyfikacja typu wydarzenia
├── event_extractor.py            # Główny pipeline
└── datasets/
    ├── training_data.csv         # Zbiór treningowy (220+ przykładów)
    └── test_relations.csv        # Zbiór testowy (105+ przykładów)
```

## Zbiory danych

### Zbiór treningowy (`training_data.csv`)
- **Format**: CSV z kolumnami `sentence`, `label`
- **Rozmiar**: 220+ przykładów
- **Kategorie**: PRZESTĘPSTWO, WYPADEK, POŻAR, POLITYKA, SPORT, EKONOMIA, NAUKA, KLĘSKA_ŻYWIOŁOWA, MEDYCYNA, PROTESTY, KULTURA, TECHNOLOGIA, PRAWO, BEZPIECZEŃSTWO, ADMINISTRACJA, SPOŁECZEŃSTWO, INFRASTRUKTURA, EKOLOGIA, EDUKACJA

### Zbiór testowy (`test_relations.csv`)
- **Format**: CSV z kolumnami `sentence`, `who`, `trigger`, `what`
- **Rozmiar**: 105+ przykładów z oznaczeniami relacji

## Rozszerzanie systemu

### Dodawanie nowych przykładów treningowych

Edytuj plik `datasets/training_data.csv`:
```csv
sentence,label
Nowe zdanie do klasyfikacji.,KATEGORIA
```

### Dodawanie nowych przykładów testowych

Edytuj plik `datasets/test_relations.csv`:
```csv
sentence,who,trigger,what
Nowe zdanie testowe.,podmiot,czasownik,dopełnienie
```

### Zapisywanie i wczytywanie modelu

```python
# Zapisz wytrenowany model
extractor.save_classifier("moj_model.pkl")

# Wczytaj model
extractor.load_classifier("moj_model.pkl")
```

## Technologie

- **spaCy** (3.7+) - NLP i Universal Dependencies parsing
- **scikit-learn** (1.3+) - Klasyfikacja ML
- **pandas** (2.0+) - Manipulacja danymi

## Wymagania systemowe

- Python 3.8 lub nowszy
- 4GB RAM (minimum)
- 2GB wolnej przestrzeni dyskowej (dla modeli)

## Autorzy

Projekt stworzony w ramach kursu NLP.

## Licencja

Ten projekt jest dostępny na licencji MIT.
