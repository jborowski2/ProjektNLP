from typing import Optional, Tuple

from compat import ensure_supported_python

ensure_supported_python()

import spacy


class RelationExtractor:
    def __init__(self, model_name: str = "pl_core_news_lg"):
        self.nlp = spacy.load(model_name)

        # spaCy Polish pipelines often use lowercase labels like: persName, geogName, placeName, date
        # Keep compatibility with pipelines that use LOC/GPE/DATE/TIME.
        self._ner_person_labels = {"persName", "PER", "PERSON"}
        self._ner_location_labels = {"geogName", "placeName", "LOC", "GPE"}
        self._ner_time_labels = {"date", "DATE", "time", "TIME"}

        # Heuristics: common Polish place nouns often aren't tagged as LOC/GPE by NER
        # (e.g., "w lesie", "na ulicy"). Using lemmas keeps it robust to inflection.
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

        # Common Polish prepositions that often introduce a location phrase.
        # Used as a fallback when NER does not mark a city/town name.
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

        # Prepositions that commonly introduce time phrases in Polish.
        # (Keep it separate from locations; some overlap is OK.)
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

        # Verbs that often appear in reporting / attribution clauses after a dash.
        # We prefer a more "content" trigger when available (e.g., "stworzy" over "poinformowała").
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
        who = None
        what = None
        where = None
        when = None
        trigger = None

        subject_phrase = None
        subject_token = None

        trigger_token = self._pick_trigger_token(doc)
        if trigger_token is None:
            # No trigger found; still try fallbacks for location/time.
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

        # Fallbacks: if UD attachments miss a location/time, use NER + preposition.
        if where is None:
            where = self._fallback_where(doc)
        if when is None:
            when = self._fallback_when(doc)

        # If we still don't have WHERE, but the subject itself is a location (common in tagged data), reuse it.
        if where is None and subject_phrase is not None:
            subj_token = next((c for c in trigger_token.children if c.dep_.split(":", 1)[0] == "nsubj"), None)
            if subj_token is not None and self._looks_like_location(subj_token):
                where = subject_phrase

        # WHAT fallback: many headlines put the "thing" as a subject (especially when WHO is a place/org).
        if what is None and subject_phrase:
            subj_token = next((c for c in trigger_token.children if c.dep_.split(":", 1)[0] == "nsubj"), None)
            if subj_token is None:
                what = subject_phrase
            else:
                if not self._looks_like_location(subj_token) and not self._is_time(subj_token):
                    what = subject_phrase

        # Heuristic: economic/stat headlines often look like:
        #   "Inflacja w Holandii przekroczyła 10 proc." where tagged: WHO=Holandia, WHAT=inflacja.
        # If the grammatical subject is a common noun and we have a location phrase,
        # prefer subject as WHAT and location as WHO when WHAT currently looks numeric-like.
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

        doc = self.nlp(sentence)
        return self.extract_relations_from_doc(doc)

    # -------------------------------------------------

    def _looks_like_location(self, token) -> bool:
        # Disambiguation: don't treat time phrases like locations.
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

        # Fallback: preposition + proper noun often means a place name
        # e.g. "w Warszawie", "pod Poznaniem", "na Śląsku".
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
        # 1) Prefer a location entity that sits in a prepositional phrase (e.g. "w Polsce").
        loc_ents = [e for e in doc.ents if e.label_ in self._ner_location_labels]
        if loc_ents:
            best = None
            best_score = -1
            for ent in loc_ents:
                start = self._expand_left_for_prep(doc, ent.start, allowed_preps=self._location_preps, max_lookback=4)
                score = 0
                if start < ent.start:
                    score += 2
                # Slight preference for later entities (often "w X" comes after subject)
                score += int(ent.start)
                if score > best_score:
                    best_score = score
                    best = (start, ent.end)
            if best is not None:
                s, e = best
                return doc[s:e].text

        # 2) If NER missed it, look for preposition-led location phrases anywhere in the sentence.
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"} and self._looks_like_location(token):
                has_loc_prep = any(
                    (t.pos_ == "ADP" or t.dep_.split(":", 1)[0] == "case")
                    and t.lemma_.lower() in self._location_preps
                    for t in token.subtree
                )
                if not has_loc_prep:
                    continue
                # Avoid time-like phrases such as "w poniedziałek".
                if self._is_time(token):
                    continue
                return self._full_phrase(token)

        return None

    def _fallback_when(self, doc) -> Optional[str]:
        # Prefer explicit date/time entities; otherwise common time words.
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

            # Fallback when NER misses dates like "w poniedziałek", "w lutym".
            if t.lemma_.lower() in self._day_lemmas or t.lemma_.lower() in self._month_lemmas:
                start = self._expand_left_for_prep(doc, t.i, allowed_preps=self._time_preps, max_lookback=3)
                return doc[start: t.i + 1].text
        return None

    def _clean_object(self, token) -> str:
        return self._full_phrase(token)

    def _full_phrase(self, token) -> str:
        # Preserve surface form as in the original sentence (punctuation + whitespace).
        parts = [t.text_with_ws for t in sorted(token.subtree, key=lambda t: t.i)]
        return "".join(parts).strip()

    def _expand_left_for_prep(self, doc, start_i: int, *, allowed_preps: set[str], max_lookback: int) -> int:
        """Try to include a nearby preceding preposition in the returned span.

        Examples:
        - "w pięknej Polsce"  -> include "w"
        - "w zeszły poniedziałek" -> include "w"

        We scan a small window to the left because modifiers (ADJ/NUM/DET) can sit
        between the preposition and the entity head.
        """
        if start_i <= 0:
            return start_i

        # Scan a small window to the left; stop early on punctuation or clause boundaries.
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

        # If there's an attribution clause after a dash, tagged TRIGGER often comes from that clause.
        dash_i = next((t.i for t in tokens if t.text in {"–", "-", "—"}), None)
        if dash_i is not None:
            for t in tokens:
                if t.i <= dash_i:
                    continue
                if t.pos_ in {"VERB", "AUX"} and ("Fin" in t.morph.get("VerbForm")):
                    # Prefer complement for AUX, e.g., "został potwierdzony" -> "potwierdzić"
                    if t.pos_ == "AUX":
                        for child in t.children:
                            dep = child.dep_.split(":", 1)[0]
                            if dep in {"xcomp", "ccomp", "acl"} and child.pos_ in {"VERB", "ADJ"}:
                                return child
                    return t

        # Otherwise use the syntactic ROOT verb if present.
        root = next((t for t in tokens if t.dep_ == "ROOT"), None)
        if root is not None and root.pos_ in {"VERB", "AUX"}:
            if root.pos_ == "AUX":
                for child in root.children:
                    dep = child.dep_.split(":", 1)[0]
                    if dep in {"xcomp", "ccomp", "acl"} and child.pos_ in {"VERB", "ADJ"}:
                        return child
            return root

        # Next best: first verb.
        for t in tokens:
            if t.pos_ == "VERB":
                return t

        # If no verb exists, use ROOT NOUN (headline nominal triggers: "wstrzymanie", "wzrost").
        if root is not None and root.pos_ in {"NOUN", "PROPN"}:
            return root

        return None
