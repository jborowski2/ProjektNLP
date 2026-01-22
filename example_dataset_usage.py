#!/usr/bin/env python3
"""
Example script demonstrating the usage of the extended dataset.
Shows examples from all categories including BRAK_ZDARZENIA.
"""

import pandas as pd
import random

def show_dataset_examples():
    """Show examples from the extended dataset"""
    
    print("=" * 70)
    print("Extended Dataset Examples - ProjektNLP")
    print("=" * 70)
    
    # Load dataset
    df = pd.read_csv('datasets/training_data.csv')
    
    print(f"\nDataset size: {len(df)} examples")
    print(f"Number of categories: {df['label'].nunique()}")
    
    # Get all unique labels
    labels = sorted(df['label'].unique())
    
    print("\n" + "=" * 70)
    print("Sample sentences from each category:")
    print("=" * 70)
    
    for label in labels:
        # Get examples for this label
        examples = df[df['label'] == label]['sentence'].tolist()
        
        # Highlight BRAK_ZDARZENIA
        if label == 'BRAK_ZDARZENIA':
            print(f"\n*** {label} (NEW CATEGORY) ***")
        else:
            print(f"\n{label}")
        
        print("-" * 70)
        
        # Show 2 random examples
        sample_size = min(2, len(examples))
        samples = random.sample(examples, sample_size)
        
        for i, example in enumerate(samples, 1):
            # Truncate long sentences
            if len(example) > 65:
                example = example[:62] + "..."
            print(f"  {i}. {example}")
    
    # Show statistics for BRAK_ZDARZENIA
    print("\n" + "=" * 70)
    print("BRAK_ZDARZENIA Category Details")
    print("=" * 70)
    
    brak_examples = df[df['label'] == 'BRAK_ZDARZENIA']
    print(f"\nNumber of examples: {len(brak_examples)}")
    print(f"Percentage of dataset: {len(brak_examples)/len(df)*100:.1f}%")
    
    print("\nWhat is BRAK_ZDARZENIA?")
    print("-" * 70)
    print("This category contains neutral/descriptive sentences that do not")
    print("describe specific events. These are important for:")
    print("  • Filtering irrelevant sentences in news processing")
    print("  • Improving classifier precision by learning to reject non-events")
    print("  • More realistic system behavior on full articles")
    
    print("\nExample sentences from BRAK_ZDARZENIA:")
    print("-" * 70)
    for i, sentence in enumerate(brak_examples['sentence'].head(5), 1):
        print(f"  {i}. {sentence}")
    
    # Compare with event categories
    print("\n" + "=" * 70)
    print("Comparison: Events vs. Non-Events")
    print("=" * 70)
    
    comparisons = [
        ("BRAK_ZDARZENIA", "W parku rosną duże drzewa."),
        ("PRZESTĘPSTWO", "Złodziej ukradł samochód z parkingu."),
        ("BRAK_ZDARZENIA", "Budynek ma dziesięć pięter."),
        ("WYPADEK", "Samochód uderzył w drzewo."),
        ("BRAK_ZDARZENIA", "Dzisiaj jest piękna pogoda."),
        ("POŻAR", "Pożar strawił budynek mieszkalny."),
    ]
    
    print("\nSide-by-side examples:")
    for label, sentence in comparisons:
        marker = "→" if label == "BRAK_ZDARZENIA" else "✓"
        print(f"  {marker} [{label:20}] {sentence}")
    
    print("\n" + "=" * 70)
    print("Dataset Usage Example")
    print("=" * 70)
    
    print("""
# Load and use the dataset

import pandas as pd
from event_classifier import EventClassifier

# Load dataset
df = pd.read_csv('datasets/training_data.csv')
sentences = df['sentence'].tolist()
labels = df['label'].tolist()

# Train classifier
classifier = EventClassifier()
classifier.train(sentences, labels)

# Test predictions
test_sentences = [
    "Złodziej ukradł portfel.",          # -> PRZESTĘPSTWO
    "W parku rosną kwiaty.",             # -> BRAK_ZDARZENIA
    "Pożar zniszczył fabrykę.",          # -> POŻAR
    "Dzisiaj jest słonecznie.",          # -> BRAK_ZDARZENIA
]

for sentence in test_sentences:
    label, confidence = classifier.predict(sentence)
    print(f"{sentence} -> {label} ({confidence:.2f})")
""")
    
    print("=" * 70)
    print("For more information, see datasets/README.md")
    print("=" * 70)


if __name__ == "__main__":
    random.seed(42)  # For reproducible examples
    show_dataset_examples()
