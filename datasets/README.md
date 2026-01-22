# Zbiory danych dla ProjektNLP

## Zbiór treningowy - `training_data.csv`

### Opis
Zbiór danych do trenowania klasyfikatora typów wydarzeń w polskich newsach. Zawiera 1042 przykłady zdań z 20 kategorii, w tym nową kategorię BRAK_ZDARZENIA dla zdań neutralnych/opisowych bez wydarzeń.

### Format
```csv
sentence,label
Zdanie przykładowe.,KATEGORIA
```

### Rozmiar i rozkład
**Łącznie: 1042 przykłady**

| Kategoria | Liczba przykładów | Opis |
|-----------|-------------------|------|
| ADMINISTRACJA | 52 | Decyzje administracyjne, działania urzędów, samorządów |
| BEZPIECZEŃSTWO | 42 | Działania służb specjalnych, wojska, straży granicznej |
| **BRAK_ZDARZENIA** | **90** | **Zdania neutralne, opisowe bez konkretnych wydarzeń** |
| EDUKACJA | 46 | Działania szkół, uniwersytetów, nauczycieli, uczniów |
| EKOLOGIA | 49 | Działania ekologiczne, ochrona środowiska, protesty ekologiczne |
| EKONOMIA | 61 | Giełda, banki, przedsiębiorstwa, finanse, biznes |
| INFRASTRUKTURA | 51 | Budowa, modernizacja infrastruktury, inwestycje |
| KLĘSKA_ŻYWIOŁOWA | 49 | Powodzie, trzęsienia ziemi, huragany, susze |
| KULTURA | 49 | Sztuka, muzyka, teatr, literatura, festiwale |
| MEDYCYNA | 47 | Operacje, leczenie, szpitale, ratownictwo |
| NAUKA | 51 | Badania naukowe, odkrycia, eksperymenty |
| POLITYKA | 53 | Działania rządu, parlamentu, polityków |
| POŻAR | 47 | Pożary budynków, lasów, przemysłu |
| PRAWO | 55 | Wyroki, procesy sądowe, działania prokuratury |
| PROTESTY | 52 | Manifestacje, strajki, demonstracje |
| PRZESTĘPSTWO | 46 | Kradzieże, napady, włamania, przestępstwa |
| SPORT | 48 | Mecze, zawody, osiągnięcia sportowe |
| SPOŁECZEŃSTWO | 50 | Działania organizacji, wolontariat, inicjatywy obywatelskie |
| TECHNOLOGIA | 47 | Nowe technologie, produkty, innowacje IT |
| WYPADEK | 57 | Wypadki drogowe, kolizje, katastrofy |

### Kategoria BRAK_ZDARZENIA

Nowa kategoria wprowadzona do rozróżniania zdań zawierających opisy wydarzeń od zdań neutralnych/opisowych. Przykłady:

```
Dzisiaj jest piękna pogoda. -> BRAK_ZDARZENIA
W parku rosną duże drzewa. -> BRAK_ZDARZENIA
Budynek ma dziesięć pięter. -> BRAK_ZDARZENIA
Sklep jest otwarty od rana. -> BRAK_ZDARZENIA
```

Kategoria ta jest istotna dla:
- Filtrowania zdań nieistotnych w przetwarzaniu newsów
- Poprawy precyzji klasyfikatora poprzez uczenie się odrzucania zdań bez wydarzeń
- Bardziej realistycznego działania systemu na pełnych artykułach (nie każde zdanie to wydarzenie)

### Źródło danych

Zbiór danych został stworzony specjalnie dla tego projektu i zawiera:

1. **Oryginalny zbiór** (220 przykładów) - `training_data_original.csv`
   - Ręcznie utworzone przykłady bazujące na typowych wzorcach polskich newsów

2. **Rozszerzony zbiór** (1042 przykłady) - `training_data.csv` i `training_data_extended.csv`
   - Syntetyczne przykłady wygenerowane dla każdej kategorii
   - Przykłady wzorowane na rzeczywistych konstrukcjach językowych polskich newsów
   - Dodano nową kategorię BRAK_ZDARZENIA (90 przykładów)
   - Minimum 42 przykłady na kategorię dla zbalansowanego treningu

### Skrypty generujące

- `generate_extended_dataset.py` - Początkowy skrypt generujący podstawowy rozszerzony zbiór
- `create_final_dataset.py` - Finalny skrypt tworzący pełny zbiór 1000+ przykładów

### Użycie

```python
import pandas as pd

# Wczytaj zbiór treningowy
df = pd.read_csv('datasets/training_data.csv')

# Przygotuj dane do treningu
sentences = df['sentence'].tolist()
labels = df['label'].tolist()

# Trenuj klasyfikator
from event_classifier import EventClassifier
classifier = EventClassifier()
classifier.train(sentences, labels)
```

## Zbiór testowy - `test_relations.csv`

### Opis
Zbiór danych do testowania ekstrakcji relacji (KTO-CO-Trigger) z wykorzystaniem Universal Dependencies.

### Format
```csv
sentence,who,trigger,what
Zdanie testowe.,podmiot,czasownik,dopełnienie
```

### Rozmiar
- **105+ przykładów** z oznaczonymi relacjami

### Użycie

```python
import pandas as pd
from relation_extractor import RelationExtractor

# Wczytaj dane testowe
df = pd.read_csv('datasets/test_relations.csv')

# Testuj ekstrakcję relacji
extractor = RelationExtractor()
for _, row in df.iterrows():
    sentence = row['sentence']
    who, trigger, what = extractor.extract_who_what(sentence)
    print(f"Sentence: {sentence}")
    print(f"Expected: {row['who']} - {row['trigger']} - {row['what']}")
    print(f"Got: {who} - {trigger} - {what}\n")
```

## Rozszerzanie zbiorów danych

### Dodawanie nowych przykładów do zbioru treningowego

Edytuj plik `datasets/training_data.csv` i dodaj nowe wiersze:

```csv
Nowe zdanie o wydarzeniu.,KATEGORIA
Kolejne zdanie opisowe.,BRAK_ZDARZENIA
```

### Dodawanie nowych przykładów do zbioru testowego

Edytuj plik `datasets/test_relations.csv` i dodaj nowe wiersze:

```csv
Nowe zdanie testowe.,kto,akcja,co
```

### Wskazówki dotyczące tworzenia przykładów

1. **BRAK_ZDARZENIA** - Używaj dla:
   - Opisów statycznych (np. "Budynek ma pięć pięter")
   - Stwierdzeń ogólnych (np. "Pogoda jest słoneczna")
   - Opisów przedmiotów/miejsc (np. "W parku rosną drzewa")

2. **Kategorie wydarzeń** - Używaj dla:
   - Zdań z akcją/działaniem (np. "Policja zatrzymała podejrzanego")
   - Zdań opisujących konkretne wydarzenie (np. "Pożar zniszczył budynek")
   - Zdań informujących o zmianach (np. "Minister ogłosił reformę")

3. **Zachowaj balans** - Staraj się mieć podobną liczbę przykładów w każdej kategorii

4. **Używaj naturalnego języka** - Wzoruj się na rzeczywistych newsach i artykułach prasowych

## Licencja

Zbiory danych są dostępne na licencji MIT jako część projektu ProjektNLP.
