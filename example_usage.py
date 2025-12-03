#!/usr/bin/env python3
"""
Example usage of the Event Extraction system.

This demonstrates how to use all components together to extract
structured event information from Polish news text.

Note: Requires models to be downloaded first:
    python setup_models.py
    or
    python -m spacy download pl_core_news_lg
"""

from event_extractor import EventExtractor


def example_single_sentence():
    """Example 1: Extract event from a single sentence"""
    print("=" * 70)
    print("Example 1: Single Sentence Extraction")
    print("=" * 70)
    
    # Initialize the extractor
    extractor = EventExtractor()
    
    # Train the classifier
    print("\nTraining classifier on provided dataset...")
    extractor.train_classifier("datasets/training_data.csv")
    
    # Extract event from the example sentence from problem statement
    sentence = "Napastnik pobił ochroniarza przed klubem."
    print(f"\nInput sentence: {sentence}")
    print("-" * 70)
    
    event = extractor.extract_event(sentence)
    if event:
        print(event)
    else:
        print("Could not extract event from sentence.")
    
    return extractor


def example_multiple_sentences(extractor):
    """Example 2: Extract events from multiple sentences"""
    print("\n" + "=" * 70)
    print("Example 2: Multiple Sentence Extraction")
    print("=" * 70)
    
    # News article with multiple sentences
    news_text = """
    Napastnik pobił ochroniarza przed klubem nocnym w centrum miasta.
    Policja zatrzymała podejrzanego dwie godziny później.
    Ochroniarz trafił do szpitala z poważnymi obrażeniami.
    Sprawca usłyszy zarzuty ciężkiego pobicia.
    """
    
    print(f"\nInput text:\n{news_text}")
    print("-" * 70)
    
    events = extractor.extract_events_from_text(news_text)
    print(f"\nExtracted {len(events)} events:")
    print("=" * 70)
    
    for i, event in enumerate(events, 1):
        print(f"\nEvent {i}:")
        print(event)
        print("-" * 70)


def example_different_categories(extractor):
    """Example 3: Test different event categories"""
    print("\n" + "=" * 70)
    print("Example 3: Different Event Categories")
    print("=" * 70)
    
    test_sentences = [
        ("Samochód uderzył w drzewo na autostradzie.", "WYPADEK"),
        ("Premier ogłosił nowe przepisy podatkowe.", "POLITYKA"),
        ("Piłkarz strzelił hat-tricka w finałowym meczu.", "SPORT"),
        ("Naukowcy odkryli nowy gatunek owadów.", "NAUKA"),
        ("Pożar zniszczył budynek mieszkalny.", "POŻAR"),
        ("Firma ogłosiła bankructwo.", "EKONOMIA"),
    ]
    
    print("\nTesting classification and extraction on various categories:")
    print("=" * 70)
    
    for sentence, expected_category in test_sentences:
        print(f"\nSentence: {sentence}")
        print(f"Expected category: {expected_category}")
        print("-" * 70)
        
        event = extractor.extract_event(sentence)
        if event:
            print(event)
            match = "✓" if event.event_type == expected_category else "✗"
            print(f"Classification match: {match}")
        else:
            print("Could not extract event.")
        print()


def example_evaluation(extractor):
    """Example 4: Evaluate on test dataset"""
    print("\n" + "=" * 70)
    print("Example 4: Evaluation on Test Dataset")
    print("=" * 70)
    
    metrics = extractor.evaluate_relation_extraction("datasets/test_relations.csv")
    
    print("\n" + "=" * 70)
    print("Evaluation Summary:")
    print("=" * 70)
    print(f"WHO Accuracy: {metrics['who_accuracy']*100:.2f}%")
    print(f"Trigger Accuracy: {metrics['trigger_accuracy']*100:.2f}%")
    print(f"WHAT Accuracy: {metrics['what_accuracy']*100:.2f}%")
    print(f"Overall Accuracy: {metrics['overall_accuracy']*100:.2f}%")
    print(f"Total Test Samples: {metrics['total_samples']}")


def example_save_load_model(extractor):
    """Example 5: Save and load trained model"""
    print("\n" + "=" * 70)
    print("Example 5: Save and Load Model")
    print("=" * 70)
    
    # Save the trained model
    model_path = "trained_event_classifier.pkl"
    print(f"\nSaving trained model to: {model_path}")
    extractor.save_classifier(model_path)
    
    # Create a new extractor and load the model
    print("\nCreating new extractor and loading saved model...")
    new_extractor = EventExtractor()
    new_extractor.load_classifier(model_path)
    
    # Test the loaded model
    test_sentence = "Kierowca potrącił pieszego na przejściu."
    print(f"\nTesting loaded model with: {test_sentence}")
    print("-" * 70)
    
    event = new_extractor.extract_event(test_sentence)
    if event:
        print(event)
    
    print("\nModel successfully saved and loaded!")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("EVENT EXTRACTION SYSTEM - COMPLETE EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates the full capabilities of the system.")
    print("It will:")
    print("  1. Train the event classifier on 220+ examples")
    print("  2. Extract events from single and multiple sentences")
    print("  3. Test different event categories")
    print("  4. Evaluate relation extraction accuracy")
    print("  5. Save and load trained models")
    print("\nNote: First run may take a few minutes to load models.")
    print("=" * 70)
    
    try:
        # Example 1: Single sentence
        extractor = example_single_sentence()
        
        # Example 2: Multiple sentences
        example_multiple_sentences(extractor)
        
        # Example 3: Different categories
        example_different_categories(extractor)
        
        # Example 4: Evaluation
        example_evaluation(extractor)
        
        # Example 5: Save/Load
        example_save_load_model(extractor)
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nNext steps:")
        print("  - Add more training examples to improve accuracy")
        print("  - Extend the system to extract WHERE and WHEN")
        print("  - Integrate with real news sources (RSS, APIs)")
        print("  - Create a web interface for easy usage")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Installed requirements: pip install -r requirements.txt")
        print("  2. Downloaded models: python setup_models.py")
        print("=" * 70)


if __name__ == "__main__":
    main()
