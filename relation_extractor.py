import spacy
from typing import Optional, Tuple


class RelationExtractor:
    def __init__(self, model_name: str = "pl_core_news_lg"):
        self.nlp = spacy.load(model_name)

    def extract_relations(
        self, sentence: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str],
               Optional[str], Optional[str]]:

        doc = self.nlp(sentence)

        who = None
        what = None
        where = None
        when = None
        trigger = None

        for token in doc:
            if token.pos_ == "VERB":
                trigger = token.lemma_

                for child in token.children:

                    if child.dep_ == "nsubj":
                        who = self._full_phrase(child)

                    elif child.dep_ == "obj":
                        what = self._clean_object(child)

                    elif child.dep_ == "obl":
                        phrase = self._full_phrase(child)

                        # GDZIE
                        if self._is_location(child) and where is None:
                            where = phrase

                        # KIEDY
                        elif self._is_time(child):
                            when = phrase

                        elif what is None:
                            what = phrase

                break

        return who, trigger, what, where, when

    # -------------------------------------------------

    def _is_location(self, token) -> bool:
        if token.ent_type_ in {"LOC", "GPE"}:
            return True

        return any(
            t.ent_type_ in {"LOC", "GPE"}
            for t in token.subtree
        )

    def _is_time(self, token) -> bool:
        time_words = {"dziś", "wczoraj", "jutro", "rano", "wieczorem"}
        return token.ent_type_ == "DATE" or token.text.lower() in time_words

    def _clean_object(self, token) -> str:
        return token.text

    def _full_phrase(self, token) -> str:
        return " ".join(
            t.text for t in sorted(token.subtree, key=lambda t: t.i)
        )
