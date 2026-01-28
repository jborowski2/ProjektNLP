# Quick Start Guide - ProjektNLP

## Szybki Start

Projekt działa na spaCy, więc do ekstrakcji relacji potrzebujesz modelu językowego. Najszybciej zacząć od instalacji zależności i pobrania modelu spaCy.

## Instalacja pełnego systemu

### Krok 1: Zainstaluj zależności
```bash
pip install -r requirements.txt
```

To pobierze:
- spaCy (framework NLP)
- scikit-learn (klasyfikacja)
- pandas (dane)

**Uwaga**: Instalacja może zająć kilka minut i wymaga około 2GB przestrzeni.

### Krok 2: Pobierz model spaCy
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

# Trenowanie (join po id z dwóch plików)
extractor.train(
    "datasets/id_and_headline_first_sentence (1).csv",
    "datasets/tagged.csv",
)

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

for line in text.strip().splitlines():
    line = line.strip()
    if not line:
        continue
    event = extractor.extract_event(line)
    print(event)
    print("-" * 40)
```

### Zapis i odczyt modelu

```python
# Trenuj raz
extractor.train(
    "datasets/id_and_headline_first_sentence (1).csv",
    "datasets/tagged.csv",
)
extractor.save_classifier("moj_model.joblib")

# Potem wczytuj bez trenowania
extractor2 = EventExtractor()
extractor2.load_classifier("moj_model.joblib")
event = extractor2.extract_event("Premier ogłosił nowe przepisy.")
```

## Testowanie komponentów osobno

### Test ekstrakcji relacji

```python
from relation_extractor import RelationExtractor

extractor = RelationExtractor()
who, trigger, what, where, when = extractor.extract_relations("Kierowca potrącił pieszego.")
print(f"KTO: {who}")      # kierowca
print(f"Trigger: {trigger}")  # potrącić
print(f"CO: {what}")       # pieszego
```

### Test klasyfikatora

```python
from event_classifier import EventClassifier
from data_loading import load_event_type_training_frame

classifier = EventClassifier()

# Wczytaj dane (join po id z dwóch plików)
df = load_event_type_training_frame(
    headlines_csv_path="datasets/id_and_headline_first_sentence (1).csv",
    tagged_csv_path="datasets/tagged.csv",
)

# Trenuj
classifier.train(df["sentence"].tolist(), df["label"].tolist())

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

### Problem: błędy instalacji / importu na Python 3.14

Na Windows najczęściej dzieje się to przy Pythonie 3.14 — spaCy (i zależności wokół Pydantic v1) mogą nie być jeszcze kompatybilne i import kończy się błędem.

**Rozwiązanie (zalecane):** użyj Pythona 3.12 (x64), usuń `.venv`, utwórz venv od nowa i uruchom `pip install -r requirements.txt`.

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
- `sentence`: Oryginalne zdanie (str)

## Dalsze kroki

1. **Rozbuduj zbiór treningowy**: Dodaj/uzupełnij oznaczenia w `datasets/tagged.csv` (oraz zadbaj o zgodne `id` w pliku z nagłówkami)
2. **Dodaj więcej kategorii**: Edytuj istniejące kategorie lub dodaj nowe
3. **Popraw ekstrakcję lokalizacji i czasu**: Rozszerz `RelationExtractor` o wykrywanie WHERE/WHEN
4. **Zintegruj z realnym źródłem danych**: Podłącz RSS/API z newsami

## Dodatkowe zasoby

- [Dokumentacja spaCy](https://spacy.io/usage)
- [Universal Dependencies](https://universaldependencies.org/)
