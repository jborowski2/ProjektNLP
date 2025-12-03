#!/usr/bin/env python3
"""
Relation extraction using Universal Dependencies.
Extracts WHO (subject) and WHAT (object) from Polish sentences.
"""

import spacy
from typing import Optional, Tuple, List
from event_record import EventRecord


class RelationExtractor:
    """
    Extracts relations from sentences using Universal Dependencies parsing.
    
    Uses spaCy with Polish language model to identify:
    - WHO (X): nsubj(V, X) - nominal subject
    - WHAT: obj(V, X) or obl(V, X) - direct object or oblique nominal
    """
    
    def __init__(self, model_name: str = "pl_core_news_lg"):
        """
        Initialize the relation extractor.
        
        Args:
            model_name: Name of the spaCy model to use
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise OSError(
                f"Model '{model_name}' not found. "
                f"Please run: python -m spacy download {model_name}"
            )
    
    def extract_relations(self, sentence: str) -> List[Tuple[str, str, str]]:
        """
        Extract subject-verb-object triples from a sentence.
        
        Args:
            sentence: Input sentence in Polish
            
        Returns:
            List of tuples (subject, verb, object)
        """
        doc = self.nlp(sentence)
        relations = []
        
        # Find all verbs in the sentence
        for token in doc:
            if token.pos_ == "VERB":
                subject = None
                obj = None
                
                # Look for subject (nsubj)
                for child in token.children:
                    if child.dep_ == "nsubj":
                        subject = self._get_full_phrase(child)
                        break
                
                # Look for object (obj or obl)
                for child in token.children:
                    if child.dep_ in ["obj", "obl"]:
                        obj = self._get_full_phrase(child)
                        break
                
                if subject and obj:
                    relations.append((subject, token.lemma_, obj))
        
        return relations
    
    def _get_full_phrase(self, token) -> str:
        """
        Get the full phrase for a token including its modifiers.
        
        Args:
            token: spaCy token
            
        Returns:
            Full phrase as string
        """
        # Get the subtree (token + all its children)
        subtree = list(token.subtree)
        # Sort by position in sentence
        subtree.sort(key=lambda t: t.i)
        # Return the text
        return " ".join([t.text for t in subtree])
    
    def extract_who_what(self, sentence: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extract WHO (subject), trigger (verb), and WHAT (object) from sentence.
        
        Args:
            sentence: Input sentence in Polish
            
        Returns:
            Tuple of (who, trigger, what) or (None, None, None) if not found
        """
        relations = self.extract_relations(sentence)
        
        if relations:
            # Return the first relation found
            who, trigger, what = relations[0]
            return who, trigger, what
        
        return None, None, None
    
    def analyze_sentence(self, sentence: str) -> dict:
        """
        Perform detailed dependency analysis of a sentence.
        
        Args:
            sentence: Input sentence in Polish
            
        Returns:
            Dictionary with dependency information
        """
        doc = self.nlp(sentence)
        
        analysis = {
            "tokens": [],
            "dependencies": []
        }
        
        for token in doc:
            analysis["tokens"].append({
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "dep": token.dep_,
                "head": token.head.text
            })
            
            analysis["dependencies"].append(
                f"{token.dep_}({token.head.text}, {token.text})"
            )
        
        return analysis


def test_relation_extractor():
    """Test the relation extractor with example sentences"""
    extractor = RelationExtractor()
    
    test_sentences = [
        "Napastnik pobił ochroniarza przed klubem.",
        "Policjant zatrzymał złodzieja na ulicy.",
        "Kierowca uderzył pieszego na przejściu."
    ]
    
    print("=== Testing Relation Extraction ===\n")
    
    for sentence in test_sentences:
        print(f"Zdanie: {sentence}")
        who, trigger, what = extractor.extract_who_what(sentence)
        print(f"KTO: {who}")
        print(f"Trigger: {trigger}")
        print(f"CO: {what}")
        print()


if __name__ == "__main__":
    test_relation_extractor()
