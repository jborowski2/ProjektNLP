#!/usr/bin/env python3
"""
Demo script showing the EventRecord data model and basic functionality
without requiring external model downloads.
"""

from event_record import EventRecord


def demo_event_record():
    """Demonstrate the EventRecord data model"""
    print("=" * 60)
    print("Demo: EventRecord Data Model")
    print("=" * 60)
    
    # Create example event record (from problem statement)
    event = EventRecord(
        event_type="PRZESTĘPSTWO",
        who="napastnik",
        what="ochroniarza",
        trigger="pobił",
        where="przed klubem",
        when=None,
        confidence=0.95,
        raw_sentence="Napastnik pobił ochroniarza przed klubem."
    )
    
    print("\nPrzykład 1: Pełny rekord wydarzenia")
    print("-" * 60)
    print(event)
    print()
    
    # Convert to dictionary
    print("\nSerializacja do słownika:")
    print("-" * 60)
    import json
    print(json.dumps(event.to_dict(), indent=2, ensure_ascii=False))
    print()
    
    # More examples
    examples = [
        EventRecord(
            event_type="WYPADEK",
            who="samochód",
            what="w drzewo",
            trigger="uderzył",
            where=None,
            when=None,
            confidence=0.88,
            raw_sentence="Samochód uderzył w drzewo."
        ),
        EventRecord(
            event_type="POLITYKA",
            who="premier",
            what="nowe przepisy",
            trigger="ogłosił",
            where=None,
            when=None,
            confidence=0.92,
            raw_sentence="Premier ogłosił nowe przepisy."
        )
    ]
    
    print("\nPrzykład 2: Więcej wydarzeń")
    print("-" * 60)
    for i, event in enumerate(examples, 1):
        print(f"\nWydarzenie {i}:")
        print(event)
        print()


def show_project_structure():
    """Show the project structure and components"""
    print("=" * 60)
    print("Struktura Systemu Ekstrakcji Wydarzeń")
    print("=" * 60)
    print()
    print("Komponenty:")
    print("-" * 60)
    print("1. event_record.py")
    print("   └─ EventRecord: Model danych dla wydarzeń")
    print()
    print("2. relation_extractor.py")
    print("   └─ RelationExtractor: Ekstrakcja WHO/WHAT/Trigger")
    print("   └─ Wykorzystuje: Universal Dependencies (spaCy)")
    print("   └─ Reguły: nsubj(V,X) dla WHO, obj/obl(V,X) dla WHAT")
    print()
    print("3. event_classifier.py")
    print("   └─ EventClassifier: Klasyfikacja typu wydarzenia")
    print("   └─ Wykorzystuje: Sentence embeddings + ML")
    print("   └─ Zbiór treningowy: 220+ przykładów (rozszerzalny do 1000+)")
    print()
    print("4. event_extractor.py")
    print("   └─ EventExtractor: Główny pipeline łączący oba komponenty")
    print()
    print("Zbiory danych:")
    print("-" * 60)
    print("• datasets/training_data.csv: 220+ przykładów do trenowania")
    print("• datasets/test_relations.csv: 105+ przykładów do testowania")
    print()


def show_usage_example():
    """Show example usage code"""
    print("=" * 60)
    print("Przykład Użycia")
    print("=" * 60)
    print()
    print("# 1. Podstawowe użycie")
    print("-" * 60)
    print("""
from event_extractor import EventExtractor

# Inicjalizacja
extractor = EventExtractor()

# Trenowanie klasyfikatora
extractor.train_classifier("datasets/training_data.csv")

# Ekstrakcja wydarzenia
sentence = "Napastnik pobił ochroniarza przed klubem."
event = extractor.extract_event(sentence)
print(event)
    """)
    
    print("\n# 2. Ekstrakcja z wielu zdań")
    print("-" * 60)
    print("""
text = '''
Napastnik pobił ochroniarza przed klubem.
Policja zatrzymała podejrzanego godzinę później.
Ochroniarz trafił do szpitala.
'''

events = extractor.extract_events_from_text(text)
for event in events:
    print(event)
    print()
    """)
    
    print("\n# 3. Zapisywanie i wczytywanie modelu")
    print("-" * 60)
    print("""
# Zapisz wytrenowany model
extractor.save_classifier("event_classifier_model.pkl")

# Wczytaj w przyszłości
extractor.load_classifier("event_classifier_model.pkl")
    """)


def main():
    """Run all demos"""
    demo_event_record()
    print("\n")
    show_project_structure()
    print("\n")
    show_usage_example()
    
    print("\n" + "=" * 60)
    print("Aby uruchomić pełny system:")
    print("=" * 60)
    print("1. Zainstaluj zależności: pip install -r requirements.txt")
    print("2. Pobierz model spaCy: python setup_models.py")
    print("3. Uruchom główny skrypt: python event_extractor.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
