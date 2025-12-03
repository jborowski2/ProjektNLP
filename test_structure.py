#!/usr/bin/env python3
"""
Test script to verify code structure and imports without requiring models.
This can be run immediately after cloning to verify the installation.
"""

import sys


def test_imports():
    """Test that all modules can be imported"""
    print("=" * 70)
    print("Testing Module Imports")
    print("=" * 70)
    
    modules = [
        "event_record",
        "relation_extractor",
        "event_classifier",
        "event_extractor"
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}.py imports successfully")
        except ImportError as e:
            print(f"✗ {module}.py failed: {e}")
            failed.append(module)
    
    return len(failed) == 0


def test_event_record():
    """Test EventRecord data model"""
    print("\n" + "=" * 70)
    print("Testing EventRecord Data Model")
    print("=" * 70)
    
    try:
        from event_record import EventRecord
        
        # Create a test event
        event = EventRecord(
            event_type="PRZESTĘPSTWO",
            who="napastnik",
            what="ochroniarza",
            trigger="pobił",
            where="przed klubem",
            confidence=0.95,
            raw_sentence="Napastnik pobił ochroniarza przed klubem."
        )
        
        print(f"✓ EventRecord created successfully")
        print(f"✓ String representation works: {len(str(event))} chars")
        print(f"✓ Dict conversion works: {len(event.to_dict())} fields")
        
        # Test all fields
        assert event.event_type == "PRZESTĘPSTWO"
        assert event.who == "napastnik"
        assert event.what == "ochroniarza"
        assert event.trigger == "pobił"
        assert event.where == "przed klubem"
        assert event.confidence == 0.95
        
        print(f"✓ All fields correctly stored")
        
        return True
        
    except Exception as e:
        print(f"✗ EventRecord test failed: {e}")
        return False


def test_datasets():
    """Test that datasets exist and are properly formatted"""
    print("\n" + "=" * 70)
    print("Testing Datasets")
    print("=" * 70)
    
    import os
    import csv
    
    datasets = [
        ("datasets/training_data.csv", ["sentence", "label"], 220),
        ("datasets/test_relations.csv", ["sentence", "who", "trigger", "what"], 106)
    ]
    
    all_passed = True
    
    for filepath, expected_columns, min_rows in datasets:
        if not os.path.exists(filepath):
            print(f"✗ {filepath} not found")
            all_passed = False
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                # Check columns
                if reader.fieldnames != expected_columns:
                    print(f"✗ {filepath}: unexpected columns {reader.fieldnames}")
                    all_passed = False
                    continue
                
                # Check row count
                if len(rows) < min_rows:
                    print(f"✗ {filepath}: only {len(rows)} rows (expected {min_rows}+)")
                    all_passed = False
                    continue
                
                print(f"✓ {filepath}: {len(rows)} rows with correct columns")
                
        except Exception as e:
            print(f"✗ {filepath}: error reading file - {e}")
            all_passed = False
    
    return all_passed


def test_dependencies():
    """Test that required dependencies can be imported"""
    print("\n" + "=" * 70)
    print("Testing Dependencies (will skip if not installed)")
    print("=" * 70)
    
    dependencies = {
        "pandas": "Data manipulation",
        "numpy": "Numerical operations",
        "sklearn": "Machine learning (scikit-learn)",
    }
    
    installed = []
    missing = []
    
    for module, description in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {module} ({description})")
            installed.append(module)
        except ImportError:
            print(f"⚠ {module} ({description}) - not installed")
            missing.append(module)
    
    print(f"\nInstalled: {len(installed)}/{len(dependencies)}")
    if missing:
        print(f"Missing: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
    
    return len(missing) == 0


def test_optional_dependencies():
    """Test optional dependencies (spaCy and sentence-transformers)"""
    print("\n" + "=" * 70)
    print("Testing Optional Dependencies (needed for full functionality)")
    print("=" * 70)
    
    optional = {
        "spacy": "NLP and Universal Dependencies",
        "sentence_transformers": "Sentence embeddings",
        "torch": "PyTorch (backend for transformers)"
    }
    
    installed = []
    missing = []
    
    for module, description in optional.items():
        try:
            __import__(module)
            print(f"✓ {module} ({description})")
            installed.append(module)
        except ImportError:
            print(f"⚠ {module} ({description}) - not installed")
            missing.append(module)
    
    # Check for Polish spaCy model
    if "spacy" in installed:
        try:
            import spacy
            spacy.load("pl_core_news_lg")
            print(f"✓ pl_core_news_lg (Polish language model)")
        except OSError:
            print(f"⚠ pl_core_news_lg - not downloaded")
            print("Run: python setup_models.py")
            missing.append("pl_core_news_lg")
    
    print(f"\nInstalled: {len(installed)}/{len(optional)}")
    if missing:
        print(f"\nTo enable full functionality, install missing dependencies:")
        print("  pip install -r requirements.txt")
        if "pl_core_news_lg" in missing:
            print("  python setup_models.py")
    
    return len(missing) == 0


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("PROJECT STRUCTURE VERIFICATION")
    print("=" * 70)
    print("\nThis script verifies the project structure without running")
    print("the full pipeline (which requires model downloads).")
    print("=" * 70)
    
    results = {
        "Module Imports": test_imports(),
        "EventRecord": test_event_record(),
        "Datasets": test_datasets(),
        "Core Dependencies": test_dependencies(),
        "Optional Dependencies": test_optional_dependencies()
    }
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nThe project structure is correct!")
        print("To run the full system:")
        print("  1. pip install -r requirements.txt")
        print("  2. python setup_models.py")
        print("  3. python event_extractor.py")
    else:
        print("⚠ SOME TESTS FAILED")
        print("\nCheck the output above for details.")
        print("Most failures can be fixed by installing dependencies:")
        print("  pip install -r requirements.txt")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
