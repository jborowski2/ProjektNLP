# Quick Start Guide - ProjektNLP

## Szybki Start bez instalacji modeli

Jeśli chcesz szybko zobaczyć jak działa struktura danych bez pobierania dużych modeli, uruchom:

```bash
python demo.py
```

Ten skrypt pokazuje:
- Strukturę EventRecord
- Przykłady ekstrakcji wydarzeń
- Architekturę systemu
- Przykłady użycia API

## Instalacja pełnego systemu

### Krok 1: Zainstaluj zależności
```bash
pip install -r requirements.txt
```

To pobierze:
- spaCy (framework NLP)
- sentence-transformers (embeddingi)
- scikit-learn (klasyfikacja)
- pandas (dane)
- torch (backend dla modeli)

**Uwaga**: Instalacja może zająć kilka minut i wymaga około 2GB przestrzeni.

### Krok 2: Pobierz model spaCy
```bash
python setup_models.py
```

Lub ręcznie:
```bash
python -m spacy download pl_core_news_lg
```

Ten model zawiera:
- Parser Universal Dependencies dla języka polskiego
- Embeddingi słów
- Tagger POS (Part-of-Speech)

**Uwaga**: Model ma około 500MB.

### Krok 3: Uruchom pełny system
```bash
python event_extractor.py
```

To uruchomi:
1. Trenowanie klasyfikatora na 220+ przykładach
2. Ewaluację ekstrakcji relacji na 106+ przykładach
3. Demonstrację ekstrakcji na przykładowych zdaniach
4. Zapisanie wytrenowanego modelu

## Przykładowe użycie

### Podstawowa ekstrakcja

```python
from event_extractor import EventExtractor

# Inicjalizacja (wymaga pobranych modeli)
extractor = EventExtractor()

# Trenowanie
extractor.train_classifier("datasets/training_data.csv")

# Ekstrakcja
event = extractor.extract_event("Napastnik pobił ochroniarza przed klubem.")
print(event)
```

**Wyjście:**
```
Typ zdarzenia: PRZESTĘPSTWO
KTO: napastnik
CO: ochroniarza
Trigger: pobił
Pewność: 0.95
```

### Przetwarzanie wielu zdań

```python
text = """
Napastnik pobił ochroniarza przed klubem.
Policja zatrzymała podejrzanego.
Samochód uderzył w drzewo.
"""

events = extractor.extract_events_from_text(text)
for event in events:
    print(event)
    print("-" * 40)
```

### Zapis i odczyt modelu

```python
# Trenuj raz
extractor.train_classifier("datasets/training_data.csv")
extractor.save_classifier("moj_model.pkl")

# Potem wczytuj bez trenowania
extractor2 = EventExtractor()
extractor2.load_classifier("moj_model.pkl")
event = extractor2.extract_event("Premier ogłosił nowe przepisy.")
```

## Testowanie komponentów osobno

### Test ekstrakcji relacji

```python
from relation_extractor import RelationExtractor

extractor = RelationExtractor()
who, trigger, what = extractor.extract_who_what(
    "Kierowca potrącił pieszego."
)
print(f"KTO: {who}")      # kierowca
print(f"Trigger: {trigger}")  # potrącić
print(f"CO: {what}")       # pieszego
```

### Test klasyfikatora

```python
from event_classifier import EventClassifier
import pandas as pd

classifier = EventClassifier()

# Wczytaj dane
df = pd.read_csv("datasets/training_data.csv")
sentences = df['sentence'].tolist()
labels = df['label'].tolist()

# Trenuj
classifier.train(sentences, labels)

# Testuj
event_type, confidence = classifier.predict("Samochód wpadł do rzeki.")
print(f"Typ: {event_type}")  # WYPADEK
print(f"Pewność: {confidence:.2f}")  # 0.87
```

## Troubleshooting

### Problem: "Model 'pl_core_news_lg' not found"
**Rozwiązanie:**
```bash
python -m spacy download pl_core_news_lg
```

### Problem: Brak pamięci podczas instalacji torch
**Rozwiązanie:** Zainstaluj wersję CPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Problem: ImportError dla sentence-transformers
**Rozwiązanie:**
```bash
pip install --upgrade sentence-transformers
```

### Problem: Długie ładowanie modeli
To normalne - pierwszy raz modele są pobierane z internetu i cachowane lokalnie. Drugie uruchomienie jest znacznie szybsze.

## Struktura wyjściowa

System zwraca obiekty `EventRecord` z następującymi polami:

- `event_type`: Typ wydarzenia (str)
- `who`: Sprawca/podmiot (str)
- `what`: Obiekt/dopełnienie (str)
- `trigger`: Czasownik/akcja (str)
- `where`: Lokalizacja - opcjonalnie (str | None)
- `when`: Czas - opcjonalnie (str | None)
- `confidence`: Pewność klasyfikacji 0-1 (float)
- `raw_sentence`: Oryginalne zdanie (str)

## Dalsze kroki

1. **Rozbuduj zbiór treningowy**: Dodaj więcej przykładów do `datasets/training_data.csv`
2. **Dodaj więcej kategorii**: Edytuj istniejące kategorie lub dodaj nowe
3. **Popraw ekstrakcję lokalizacji i czasu**: Rozszerz `RelationExtractor` o wykrywanie WHERE/WHEN
4. **Zintegruj z realnym źródłem danych**: Podłącz RSS/API z newsami

## Dodatkowe zasoby

- [Dokumentacja spaCy](https://spacy.io/usage)
- [Universal Dependencies](https://universaldependencies.org/)
- [Sentence Transformers](https://www.sbert.net/)
