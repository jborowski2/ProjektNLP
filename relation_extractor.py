"""Ekstrakcja relacji (KTO/CO/GDZIE/KIEDY/TRIGGER) z pojedynczego zdania.

To nie jest model uczony, tylko heurystyki oparte o:
- analizę składniową (dependency parsing) ze spaCy,
- NER (rozpoznawanie encji),
- reguły awaryjne (prepozycje, listy lematów miejsc/czasu).

Cel: wyciągnąć prosty „szkielet zdarzenia” z nagłówków newsowych.
"""

from typing import Optional, Tuple

from compat import ensure_supported_python

ensure_supported_python()

import spacy


class RelationExtractor:
    def __init__(self, model_name: str = "pl_core_news_lg"):
        self.nlp = spacy.load(model_name)

        # Pipeline PL w spaCy często ma etykiety lowercase typu: persName, geogName, placeName, date.
        # Dla kompatybilności dopuszczamy też klasyczne LOC/GPE/DATE/TIME.
        self._ner_person_labels = {"persName", "PER", "PERSON"}
        self._ner_location_labels = {"geogName", "placeName", "LOC", "GPE"}
        self._ner_time_labels = {"date", "DATE", "time", "TIME"}

        # Heurystyka: częste rzeczowniki-miejsca często nie są oznaczane jako LOC/GPE przez NER
        # (np. "w lesie", "na ulicy"). Użycie lematów zwiększa odporność na odmianę.
        self._place_lemmas = {
            "las",
            "ulica",
            "droga",
            "autostrada",
            "skrzyżowanie",
            "most",
            "tunel",
            "miasto",
            "wieś",
            "miasteczko",
            "gmina",
            "powiat",
            "województwo",
            "region",
            "centrum",
            "park",
            "plac",
            "osiedle",
            "dzielnica",
            "okolica",
            "teren",
            "miejsce",
            "granica",
            "przejście",
            "przystanek",
            "dworzec",
            "lotnisko",
            "stacja",
            "metro",
            "stadion",
            "hala",
            "magazyn",
            "fabryka",
            "firma",
            "biuro",
            "dom",
            "mieszkanie",
            "szpital",
            "szkoła",
            "uczelnia",
            "sąd",
            "komisariat",
            "komenda",
            "posterunek",
            "hotel",
            "restauracja",
            "klub",
            "kościół",
        }

        # Typowe przyimki wprowadzające frazę miejsca.
        # Używane w fallbackach, gdy NER nie oznaczy np. nazwy miasta.
        self._location_preps = {
            "w",
            "we",
            "na",
            "nad",
            "pod",
            "przed",
            "za",
            "przy",
            "obok",
            "koło",
            "między",
            "pomiędzy",
            "u",
            "z",
            "ze",
            "do",
        }

        # Przyimki wprowadzające frazy czasu w języku polskim.
        # (Trzymamy osobno od miejsca; pewne nakładanie jest OK.)
        self._time_preps = {
            "w",
            "we",
            "na",
            "o",
            "około",
            "przed",
            "po",
            "od",
            "do",
            "między",
            "pomiędzy",
        }

        self._day_lemmas = {
            "poniedziałek",
            "wtorek",
            "środa",
            "czwartek",
            "piątek",
            "sobota",
            "niedziela",
        }

        self._month_lemmas = {
            "styczeń",
            "luty",
            "marzec",
            "kwiecień",
            "maj",
            "czerwiec",
            "lipiec",
            "sierpień",
            "wrzesień",
            "październik",
            "listopad",
            "grudzień",
        }

        # Czasowniki sprawozdawcze/atrybucyjne (często po myślniku).
        # Jeśli nagłówek ma część typu "– poinformowała X", wolimy trigger z części merytorycznej.
        self._reporting_verb_lemmas = {
            "poinformować",
            "podawać",
            "podać",
            "napisać",
            "powiedzieć",
            "dodać",
            "podkreślić",
            "zaznaczyć",
            "stwierdzić",
            "ogłosić",
            "przekazać",
            "donosić",
            "donieść",
            "alarmować",
            "ostrzec",
            "zapowiedzieć",
            "ujawnić",
            "twierdzić",
            "uważać",
        }

    def extract_relations_from_doc(
        self, doc
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Wyciągnij relacje z już sparsowanego `doc`.

        Zwracamy: (who, trigger, what, where, when).

        Ważne założenie: najpierw wybieramy token TRIGGER (czasownik/root), a potem
        próbujemy znaleźć jego argumenty (podmiot/dopełnienie/okoliczniki).
        """
        who = None
        what = None
        where = None
        when = None
        trigger = None

        subject_phrase = None
        subject_token = None

        trigger_token = self._pick_trigger_token(doc)
        if trigger_token is None:
            # Nie znaleziono TRIGGER; nadal próbujemy fallbacków dla miejsca/czasu.
            where = self._fallback_where(doc)
            when = self._fallback_when(doc)
            return who, trigger, what, where, when

        trigger = trigger_token.lemma_

        for child in trigger_token.children:
            dep = child.dep_.split(":", 1)[0]

            if dep == "nsubj":
                subject_phrase = self._full_phrase(child)
                subject_token = child

                # Prefer named people when available (improves WHO precision)
                # Jeśli w zdaniu jest encja osoby, często jest lepszym KTO niż goły rzeczownik.
                person_ents = [e.text for e in doc.ents if e.label_ in self._ner_person_labels]
                if person_ents and child.pos_ != "PROPN":
                    who = person_ents[0]
                else:
                    who = subject_phrase

            elif dep in {"obj", "iobj"}:
                if what is None:
                    what = self._clean_object(child)

            elif dep in {"xcomp", "ccomp"}:
                if what is None:
                    what = self._full_phrase(child)

            elif dep == "obl":
                phrase = self._full_phrase(child)

                # KIEDY
                if self._is_time(child) and when is None:
                    when = phrase

                # GDZIE
                elif self._looks_like_location(child) and where is None:
                    where = phrase

                elif what is None:
                    what = phrase

            elif dep == "nmod":
                # Useful for nominal triggers and many headline-style constructs.
                if what is None and (not self._is_time(child)) and (not self._looks_like_location(child)):
                    what = self._full_phrase(child)

        # Fallbacki: jeśli analiza składni nie podała miejsca/czasu, próbujemy NER + przyimki.
        if where is None:
            where = self._fallback_where(doc)
        if when is None:
            when = self._fallback_when(doc)

        # Jeśli nadal nie ma GDZIE, ale podmiot sam wygląda jak lokalizacja (częste w danych), użyj go.
        if where is None and subject_phrase is not None:
            subj_token = next((c for c in trigger_token.children if c.dep_.split(":", 1)[0] == "nsubj"), None)
            if subj_token is not None and self._looks_like_location(subj_token):
                where = subject_phrase

        # CO fallback: nagłówki często mają "co" jako podmiot (szczególnie gdy "kto" to miejsce/instytucja).
        if what is None and subject_phrase:
            subj_token = next((c for c in trigger_token.children if c.dep_.split(":", 1)[0] == "nsubj"), None)
            if subj_token is None:
                what = subject_phrase
            else:
                if not self._looks_like_location(subj_token) and not self._is_time(subj_token):
                    what = subject_phrase

        # Heurystyka dla nagłówków statystycznych/ekonomicznych, np.:
        #   "Inflacja w Holandii przekroczyła 10 proc." gdzie tagi często są: KTO=Holandia, CO=inflacja.
        # Jeśli podmiot jest rzeczownikiem pospolitym, a mamy frazę miejsca, to czasem
        # warto przepiąć: KTO := GDZIE, CO := podmiot (zwłaszcza gdy CO wygląda "liczbowo").
        if subject_phrase and subject_token is not None and where is not None:
            if who == subject_phrase and subject_token.pos_ == "NOUN" and subject_token.pos_ != "PROPN":
                if what is None or self._looks_numeric_like(what):
                    who = where
                    what = subject_phrase

        return who, trigger, what, where, when

    def _looks_numeric_like(self, text: Optional[str]) -> bool:
        if not text:
            return False
        t = text.lower()
        return any(ch.isdigit() for ch in t) or "%" in t or "proc" in t

    def extract_relations(
        self, sentence: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Wygodny wrapper: parsuje zdanie i deleguje do `extract_relations_from_doc`."""
        doc = self.nlp(sentence)
        return self.extract_relations_from_doc(doc)

    # -------------------------------------------------

    def _looks_like_location(self, token) -> bool:
        # Rozróżnienie: nie traktuj fraz czasu jako miejsca.
        if self._is_time(token):
            return False

        if token.ent_type_ in self._ner_location_labels:
            return True

        if token.lemma_.lower() in self._place_lemmas:
            return True

        for t in token.subtree:
            if t.ent_type_ in self._ner_location_labels:
                return True
            if t.lemma_.lower() in self._place_lemmas:
                return True

        # Fallback: przyimek + nazwa własna często oznacza miejsce
        # np. "w Warszawie", "pod Poznaniem", "na Śląsku".
        has_propn = any(t.pos_ == "PROPN" for t in token.subtree)
        has_loc_prep = any(
            (t.pos_ == "ADP" or t.dep_.split(":", 1)[0] == "case")
            and t.lemma_.lower() in self._location_preps
            for t in token.subtree
        )
        if has_propn and has_loc_prep:
            return True

        return False

    def _is_time(self, token) -> bool:
        time_words = {"dziś", "wczoraj", "jutro", "rano", "wieczorem"}
        if token.ent_type_ in self._ner_time_labels:
            return True

        if token.text.lower() in time_words:
            return True

        if token.lemma_.lower() in self._day_lemmas or token.lemma_.lower() in self._month_lemmas:
            return True

        for t in token.subtree:
            if t.ent_type_ in self._ner_time_labels:
                return True
            if t.lemma_.lower() in self._day_lemmas or t.lemma_.lower() in self._month_lemmas:
                return True
            if t.text.lower() in time_words:
                return True

        return False

    def _fallback_where(self, doc) -> Optional[str]:
        # 1) Preferujemy encję lokacji w obrębie frazy przyimkowej (np. "w Polsce").
        loc_ents = [e for e in doc.ents if e.label_ in self._ner_location_labels]
        if loc_ents:
            best = None
            best_score = -1
            for ent in loc_ents:
                start = self._expand_left_for_prep(doc, ent.start, allowed_preps=self._location_preps, max_lookback=4)
                score = 0
                if start < ent.start:
                    score += 2
                # Lekka preferencja dla encji później w zdaniu (często "w X" jest po podmiocie)
                score += int(ent.start)
                if score > best_score:
                    best_score = score
                    best = (start, ent.end)
            if best is not None:
                s, e = best
                return doc[s:e].text

        # 2) Jeśli NER nie trafił, szukamy fraz miejsca prowadzonych przyimkiem.
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"} and self._looks_like_location(token):
                has_loc_prep = any(
                    (t.pos_ == "ADP" or t.dep_.split(":", 1)[0] == "case")
                    and t.lemma_.lower() in self._location_preps
                    for t in token.subtree
                )
                if not has_loc_prep:
                    continue
                # Odrzuć frazy czasu typu "w poniedziałek".
                if self._is_time(token):
                    continue
                return self._full_phrase(token)

        return None

    def _fallback_when(self, doc) -> Optional[str]:
        # Preferujemy encje DATE/TIME; jeśli brak, to proste słowa czasu.
        time_ents = [e for e in doc.ents if e.label_ in self._ner_time_labels]
        if time_ents:
            ent = time_ents[0]
            start = self._expand_left_for_prep(doc, ent.start, allowed_preps=self._time_preps, max_lookback=4)
            return doc[start: ent.end].text

        time_words = {"dziś", "wczoraj", "jutro", "rano", "wieczorem"}
        for t in doc:
            if t.text.lower() in time_words:
                start = self._expand_left_for_prep(doc, t.i, allowed_preps=self._time_preps, max_lookback=2)
                return doc[start: t.i + 1].text

            # Fallback gdy NER nie złapie dat typu "w poniedziałek", "w lutym".
            if t.lemma_.lower() in self._day_lemmas or t.lemma_.lower() in self._month_lemmas:
                start = self._expand_left_for_prep(doc, t.i, allowed_preps=self._time_preps, max_lookback=3)
                return doc[start: t.i + 1].text
        return None

    def _clean_object(self, token) -> str:
        return self._full_phrase(token)

    def _full_phrase(self, token) -> str:
        # Zachowujemy formę powierzchniową jak w zdaniu (interpunkcja + spacje).
        parts = [t.text_with_ws for t in sorted(token.subtree, key=lambda t: t.i)]
        return "".join(parts).strip()

    def _expand_left_for_prep(self, doc, start_i: int, *, allowed_preps: set[str], max_lookback: int) -> int:
        """Spróbuj dołączyć przyimek stojący tuż przed frazą.

        Przykłady:
        - "w pięknej Polsce" -> dołącz "w"
        - "w zeszły poniedziałek" -> dołącz "w"

        Skanujemy małe okno w lewo, bo między przyimkiem a głową frazy mogą być
        wtrącone modyfikatory (ADJ/NUM/DET).
        """
        if start_i <= 0:
            return start_i

        # Skanujemy małe okno w lewo; stop na interpunkcji lub granicy frazy/zdania.
        stop_pos = {"PUNCT", "VERB", "AUX", "SCONJ", "CCONJ"}

        for back in range(1, max_lookback + 1):
            j = start_i - back
            if j < 0:
                break

            t = doc[j]
            if t.pos_ in stop_pos:
                break

            if t.pos_ == "ADP" and t.lemma_.lower() in allowed_preps:
                return j

        return start_i

    def _pick_trigger_token(self, doc):
        tokens = list(doc)
        if not tokens:
            return None

        # Jeśli po myślniku jest dopowiedzenie/źródło („– powiedział X”),
        # to TRIGGER w danych bywa brany z tej części.
        dash_i = next((t.i for t in tokens if t.text in {"–", "-", "—"}), None)
        if dash_i is not None:
            for t in tokens:
                if t.i <= dash_i:
                    continue
                if t.pos_ in {"VERB", "AUX"} and ("Fin" in t.morph.get("VerbForm")):
                    # Dla AUX wolimy dopełnienie, np. "został potwierdzony" -> "potwierdzony".
                    if t.pos_ == "AUX":
                        for child in t.children:
                            dep = child.dep_.split(":", 1)[0]
                            if dep in {"xcomp", "ccomp", "acl"} and child.pos_ in {"VERB", "ADJ"}:
                                return child
                    return t

        # W przeciwnym razie bierzemy syntaktyczny ROOT jeśli jest czasownikiem.
        root = next((t for t in tokens if t.dep_ == "ROOT"), None)
        if root is not None and root.pos_ in {"VERB", "AUX"}:
            if root.pos_ == "AUX":
                for child in root.children:
                    dep = child.dep_.split(":", 1)[0]
                    if dep in {"xcomp", "ccomp", "acl"} and child.pos_ in {"VERB", "ADJ"}:
                        return child
            return root

        # Kolejny fallback: pierwszy czasownik w zdaniu.
        for t in tokens:
            if t.pos_ == "VERB":
                return t

        # Jeśli nie ma czasownika, użyj ROOT NOUN (nagłówki nominalne: "wstrzymanie", "wzrost").
        if root is not None and root.pos_ in {"NOUN", "PROPN"}:
            return root

        return None
