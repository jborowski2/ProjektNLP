#!/usr/bin/env python3
"""
Main event extraction pipeline that combines relation extraction
and event classification to extract structured event records from Polish news.
"""

import pandas as pd
from typing import List, Optional
from event_record import EventRecord
from relation_extractor import RelationExtractor
from event_classifier import EventClassifier


class EventExtractor:
    """
    Complete pipeline for extracting events from Polish news articles.
    
    Combines:
    1. Universal Dependencies parsing for WHO/WHAT extraction
    2. Sentence embeddings + classification for event type
    """
    
    def __init__(
        self,
        spacy_model: str = "pl_core_news_lg",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        Initialize the event extractor.
        
        Args:
            spacy_model: Name of the spaCy model for relation extraction
            embedding_model: Name of the sentence transformer model
        """
        print("Initializing Event Extractor...")
        self.relation_extractor = RelationExtractor(spacy_model)
        self.event_classifier = EventClassifier(embedding_model)
        print("Event Extractor initialized successfully!")
    
    def train_classifier(self, training_data_path: str):
        """
        Train the event type classifier.
        
        Args:
            training_data_path: Path to CSV file with columns: sentence, label
        """
        print(f"\nLoading training data from {training_data_path}...")
        df = pd.read_csv(training_data_path)
        
        sentences = df['sentence'].tolist()
        labels = df['label'].tolist()
        
        print(f"Training on {len(sentences)} examples...")
        metrics = self.event_classifier.train(sentences, labels)
        
        return metrics
    
    def extract_event(self, sentence: str) -> Optional[EventRecord]:
        """
        Extract a complete event record from a sentence.
        
        Args:
            sentence: Input sentence in Polish
            
        Returns:
            EventRecord or None if extraction fails
        """
        # Extract WHO, trigger, WHAT using Universal Dependencies
        who, trigger, what = self.relation_extractor.extract_who_what(sentence)
        
        if not who or not trigger or not what:
            # Could not extract complete relation
            return None
        
        # Classify event type
        if self.event_classifier.is_trained:
            event_type, confidence = self.event_classifier.predict(sentence)
        else:
            event_type = "NIEZNANY"
            confidence = 0.0
        
        # Create event record
        event = EventRecord(
            event_type=event_type,
            who=who,
            what=what,
            trigger=trigger,
            confidence=confidence,
            raw_sentence=sentence
        )
        
        return event
    
    def extract_events_from_text(self, text: str) -> List[EventRecord]:
        """
        Extract all events from a multi-sentence text.
        
        Args:
            text: Input text containing multiple sentences
            
        Returns:
            List of EventRecords
        """
        # Use spaCy to split into sentences
        doc = self.relation_extractor.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        events = []
        for sentence in sentences:
            event = self.extract_event(sentence)
            if event:
                events.append(event)
        
        return events
    
    def evaluate_relation_extraction(self, test_data_path: str) -> dict:
        """
        Evaluate relation extraction on test dataset.
        
        Args:
            test_data_path: Path to CSV with columns: sentence, who, trigger, what
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating relation extraction on {test_data_path}...")
        df = pd.read_csv(test_data_path)
        
        correct_who = 0
        correct_trigger = 0
        correct_what = 0
        total = len(df)
        
        for _, row in df.iterrows():
            sentence = row['sentence']
            expected_who = str(row['who']).lower().strip()
            expected_trigger = str(row['trigger']).lower().strip()
            expected_what = str(row['what']).lower().strip()
            
            who, trigger, what = self.relation_extractor.extract_who_what(sentence)
            
            if who and expected_who in who.lower():
                correct_who += 1
            if trigger and expected_trigger in trigger.lower():
                correct_trigger += 1
            if what and expected_what in what.lower():
                correct_what += 1
        
        metrics = {
            "who_accuracy": correct_who / total,
            "trigger_accuracy": correct_trigger / total,
            "what_accuracy": correct_what / total,
            "overall_accuracy": (correct_who + correct_trigger + correct_what) / (3 * total),
            "total_samples": total
        }
        
        print("\n=== Relation Extraction Evaluation ===")
        print(f"WHO Accuracy: {metrics['who_accuracy']:.4f}")
        print(f"Trigger Accuracy: {metrics['trigger_accuracy']:.4f}")
        print(f"WHAT Accuracy: {metrics['what_accuracy']:.4f}")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"Total Samples: {metrics['total_samples']}")
        
        return metrics
    
    def save_classifier(self, filepath: str):
        """Save the trained classifier"""
        self.event_classifier.save_model(filepath)
    
    def load_classifier(self, filepath: str):
        """Load a trained classifier"""
        self.event_classifier.load_model(filepath)


def main():
    """Example usage of the EventExtractor"""
    print("=" * 60)
    print("Event Extraction from Polish News")
    print("=" * 60)
    
    # Initialize extractor
    extractor = EventExtractor()
    
    # Train classifier on training data
    print("\n" + "=" * 60)
    print("Training Event Classifier")
    print("=" * 60)
    extractor.train_classifier("datasets/training_data.csv")
    
    # Evaluate relation extraction on test data
    print("\n" + "=" * 60)
    print("Evaluating Relation Extraction")
    print("=" * 60)
    extractor.evaluate_relation_extraction("datasets/test_relations.csv")
    
    # Test on example sentences
    print("\n" + "=" * 60)
    print("Example Event Extractions")
    print("=" * 60)
    
    test_sentences = [
        "Napastnik pobił ochroniarza przed klubem.",
        "Policjant zatrzymał złodzieja na ulicy.",
        "Kierowca uderzył pieszego na przejściu.",
        "Premier ogłosił nowe przepisy.",
        "Samochód wpadł do rzeki."
    ]
    
    for sentence in test_sentences:
        print(f"\nZdanie: {sentence}")
        print("-" * 60)
        event = extractor.extract_event(sentence)
        if event:
            print(event)
        else:
            print("Nie udało się wyodrębnić wydarzenia.")
    
    # Save trained model
    print("\n" + "=" * 60)
    print("Saving trained model...")
    extractor.save_classifier("event_classifier_model.pkl")
    print("Model saved successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
