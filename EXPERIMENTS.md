# Eksperymenty: klasyfikacja typu zdarzenia

## Cel
Celem jest sprawdzenie kilku podejść (model/cechy/strategia radzenia sobie z niezbalansowaniem klas) i wybranie najlepszego według miar jakości.

## Dane
Eksperymenty korzystają z dwóch plików i łączą je po `id`:
- `datasets/id_and_headline_first_sentence (1).csv` (kolumny: `id,headline`)
- `datasets/tagged.csv` (kolumny: `id;kategoria;...`)

Etykieta (`label`) powstaje z `kategoria` (np. `PRZESTEPSTWO (zatrzymać)` → `PRZESTEPSTWO`).

## Podział danych
Standardowo stosujemy split 80/20:
- trening: 80%
- test: 20%

Przy niezbalansowanych klasach używamy `stratify`, a jeśli jest to niemożliwe (zbyt mało przykładów w jakiejś klasie), split przechodzi na tryb bez stratyfikacji.

## Miary jakości
Raportujemy co najmniej:
- `accuracy`
- `f1_macro` (najważniejsza przy niezbalansowanych klasach)
- `f1_weighted`

Dodatkowo (gdy model udostępnia `predict_proba`) raportujemy:
- `aic` i `bic` liczone z log-wiarygodności na zbiorze testowym:
	- $LL = \sum_i \log p(y_i \mid x_i)$
	- $AIC = 2k - 2LL$
	- $BIC = \log(n)\,k - 2LL$

Uwaga: dla modeli bez prawdziwych prawdopodobieństw (np. `LinearSVC`) `AIC/BIC` są ustawiane na `NaN`.

Wybór „najlepszego” podejścia: maksymalizacja `f1_macro` (średnia po kilku seedach).

## Jak uruchomić
Uruchom w aktywnym venv:

- `./.venv/Scripts/python experiments.py`

Parametry:
- `--test-size 0.2` (domyślnie)
- `--seeds 42,43,44` (domyślnie)
- `--out results/experiments.csv`

Filtrowanie (żeby szybciej testować wybrane modele):
- `--include LinearSVC,LogReg_L2`
- `--exclude XGBoost`
- `--no-xgboost` (pomija modele XGBoost)

## Wyniki
Wyniki zapisywane są do `results/experiments.csv`.
Każdy wiersz to jedna konfiguracja i jeden seed.

## Ocena ekstrakcji KTO/CO/GDZIE/KIEDY
Nie mamy ręcznie oznaczonego zbioru "gold" dla relacji, więc można uruchomić ocenę przybliżoną (proxy) na całym zbiorze:

`./.venv/Scripts/python evaluate_extraction.py`

Raporty trafiają do `results/`:
- `extraction_report.json` (podsumowanie)
- `extraction_predictions.csv` (predykcje per-zdanie)

## Modele
Domyślnie uruchamiane są m.in.:
- `LinearSVC`, `Tuned_LinearSVC`
- `LogReg_L2`, `Tuned_LogReg_L2`
- `MultinomialNB`
- `Bagging_LogReg`
- `GradientBoosting`, `MLP` (z redukcją wymiaru SVD)
- `XGBoost`, `Tuned_XGBoost` (jeśli zainstalowany `xgboost`)

### Interpretacja
- Jeśli `accuracy` jest wysokie, ale `f1_macro` niskie → model „ciągnie” do klasy większościowej.
- `f1_macro` rośnie, gdy model zaczyna trafiać klasy rzadkie.

## Dalsze kroki
- Dodać więcej przykładów dla rzadkich klas (np. `WYPADEK` ma bardzo mało obserwacji).
- Rozszerzyć siatkę eksperymentów o regularyzację (`C`) i inne n-gramy.
- (Opcjonalnie) dodać walidację krzyżową (StratifiedKFold) zamiast pojedynczego splitu.
