#!/usr/bin/env python3
"""
Event type classification using sentence embeddings.
Classifies events into categories like PRZESTĘPSTWO, WYPADEK, etc.
"""

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pickle
from typing import List, Tuple
import os


class EventClassifier:
    """
    Classifies events into types using sentence embeddings and ML.
    
    Uses sentence-transformers for Polish sentence embeddings and
    logistic regression for classification.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize the event classifier.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.is_trained = False
        self.label_names = []
    
    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Encode sentences to embeddings.
        
        Args:
            sentences: List of sentences to encode
            
        Returns:
            Numpy array of embeddings
        """
        return self.embedding_model.encode(sentences, show_progress_bar=False)
    
    def train(self, sentences: List[str], labels: List[str], test_size: float = 0.2):
        """
        Train the classifier on labeled data.
        
        Args:
            sentences: List of training sentences
            labels: List of corresponding labels
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with training metrics
        """
        # Encode sentences
        print("Encoding sentences...")
        embeddings = self.encode_sentences(sentences)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Train classifier
        print("Training classifier...")
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        self.label_names = list(set(labels))
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTraining complete!")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            "accuracy": accuracy,
            "train_size": len(X_train),
            "test_size": len(X_test)
        }
    
    def predict(self, sentence: str) -> Tuple[str, float]:
        """
        Predict the event type for a sentence.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Tuple of (predicted_label, confidence)
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction")
        
        embedding = self.encode_sentences([sentence])
        prediction = self.classifier.predict(embedding)[0]
        
        # Get confidence (probability of predicted class)
        probabilities = self.classifier.predict_proba(embedding)[0]
        confidence = max(probabilities)
        
        return prediction, confidence
    
    def save_model(self, filepath: str):
        """
        Save the trained classifier to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            "classifier": self.classifier,
            "label_names": self.label_names
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained classifier from disk.
        
        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data["classifier"]
        self.label_names = model_data["label_names"]
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")


def test_event_classifier():
    """Test the event classifier with sample data"""
    # Sample training data
    train_sentences = [
        "Napastnik pobił ochroniarza przed klubem.",
        "Złodziej ukradł samochód z parkingu.",
        "Policja zatrzymała podejrzanego o kradzież.",
        "Samochód uderzył w drzewo.",
        "Dwa pojazdy zderzyły się na skrzyżowaniu.",
        "Motocyklista stracił panowanie nad pojazdem.",
    ]
    
    train_labels = [
        "PRZESTĘPSTWO",
        "PRZESTĘPSTWO",
        "PRZESTĘPSTWO",
        "WYPADEK",
        "WYPADEK",
        "WYPADEK",
    ]
    
    print("=== Testing Event Classifier ===\n")
    
    classifier = EventClassifier()
    
    # Note: This is a minimal example. Real training would need 1000+ examples
    print("Note: This is a minimal test. Production model needs 1000+ examples.\n")
    
    classifier.train(train_sentences, train_labels, test_size=0.3)
    
    # Test predictions
    test_sentences = [
        "Kierowca potrącił pieszego.",
        "Bandyta zaatakował przechodnia."
    ]
    
    print("\n=== Test Predictions ===")
    for sentence in test_sentences:
        label, confidence = classifier.predict(sentence)
        print(f"Zdanie: {sentence}")
        print(f"Typ: {label} (pewność: {confidence:.4f})\n")


if __name__ == "__main__":
    test_event_classifier()
