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
- Wykorzystuje wektorową reprezentację zdań (sentence embeddings)
- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
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
- Python 3.8+
- pip

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

3. Pobierz model spaCy dla języka polskiego:
```bash
python setup_models.py
```

Alternatywnie można pobrać model ręcznie:
```bash
python -m spacy download pl_core_news_lg
```

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
- **sentence-transformers** (2.2+) - Embeddingi zdań
- **scikit-learn** (1.3+) - Klasyfikacja ML
- **pandas** (2.0+) - Manipulacja danymi
- **PyTorch** (2.0+) - Backend dla modeli transformerowych

## Wymagania systemowe

- Python 3.8 lub nowszy
- 4GB RAM (minimum)
- 2GB wolnej przestrzeni dyskowej (dla modeli)

## Autorzy

Projekt stworzony w ramach kursu NLP.

## Licencja

Ten projekt jest dostępny na licencji MIT.
