#!/usr/bin/env python3
"""
Simple script to verify the extended training dataset.
This script checks dataset integrity and provides statistics.
"""

import pandas as pd
import sys

def verify_dataset(filepath='datasets/training_data.csv'):
    """Verify the training dataset"""
    
    print("=" * 60)
    print("Dataset Verification Report")
    print("=" * 60)
    
    try:
        # Load dataset
        df = pd.read_csv(filepath)
        print(f"\n✓ Dataset loaded successfully from: {filepath}")
        print(f"✓ Total examples: {len(df)}")
        
        # Check columns
        expected_columns = ['sentence', 'label']
        if list(df.columns) != expected_columns:
            print(f"\n✗ ERROR: Expected columns {expected_columns}, got {list(df.columns)}")
            return False
        print(f"✓ Columns are correct: {expected_columns}")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print(f"\n✗ WARNING: Found missing values:")
            print(missing[missing > 0])
        else:
            print("✓ No missing values")
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['sentence']).sum()
        if duplicates > 0:
            print(f"\n✗ WARNING: Found {duplicates} duplicate sentences")
        else:
            print("✓ No duplicate sentences")
        
        # Get label statistics
        label_counts = df['label'].value_counts().sort_index()
        unique_labels = len(label_counts)
        
        print(f"\n✓ Number of unique labels: {unique_labels}")
        
        # Check for BRAK_ZDARZENIA
        if 'BRAK_ZDARZENIA' in label_counts.index:
            print(f"✓ BRAK_ZDARZENIA category present: {label_counts['BRAK_ZDARZENIA']} examples")
        else:
            print("✗ WARNING: BRAK_ZDARZENIA category not found")
        
        # Check minimum examples per category
        min_examples = label_counts.min()
        min_label = label_counts.idxmin()
        if min_examples < 20:
            print(f"\n✗ WARNING: Category '{min_label}' has only {min_examples} examples (recommended: 20+)")
        else:
            print(f"✓ Minimum examples per category: {min_examples} (category: {min_label})")
        
        # Display full statistics
        print("\n" + "=" * 60)
        print("Label Distribution")
        print("=" * 60)
        print(f"\n{'Label':<25} {'Count':>10} {'Percentage':>15}")
        print("-" * 60)
        
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"{label:<25} {count:>10} {percentage:>14.1f}%")
        
        print("-" * 60)
        print(f"{'TOTAL':<25} {len(df):>10} {100.0:>14.1f}%")
        
        # Sample sentences
        print("\n" + "=" * 60)
        print("Sample Sentences (5 random examples)")
        print("=" * 60)
        
        samples = df.sample(n=min(5, len(df)), random_state=42)
        for i, (_, row) in enumerate(samples.iterrows(), 1):
            sentence = row['sentence']
            label = row['label']
            # Truncate long sentences
            if len(sentence) > 60:
                sentence = sentence[:57] + "..."
            print(f"\n{i}. [{label}]")
            print(f"   {sentence}")
        
        # Check target requirements
        print("\n" + "=" * 60)
        print("Requirements Check")
        print("=" * 60)
        
        requirements = [
            ("1000+ examples", len(df) >= 1000, f"✓ {len(df)} examples"),
            ("BRAK_ZDARZENIA present", 'BRAK_ZDARZENIA' in label_counts.index, 
             f"✓ {label_counts.get('BRAK_ZDARZENIA', 0)} examples"),
            ("All categories ≥ 20 examples", min_examples >= 20, 
             f"✓ min {min_examples} examples"),
            ("No duplicates", duplicates == 0, f"✓ No duplicates"),
            ("No missing values", not missing.any(), f"✓ No missing values"),
        ]
        
        all_passed = True
        for requirement, passed, message in requirements:
            status = "✓" if passed else "✗"
            print(f"{status} {requirement}: {message}")
            if not passed:
                all_passed = False
        
        print("\n" + "=" * 60)
        if all_passed:
            print("✓ VERIFICATION PASSED - Dataset meets all requirements!")
        else:
            print("✗ VERIFICATION FAILED - Please check warnings above")
        print("=" * 60)
        
        return all_passed
        
    except FileNotFoundError:
        print(f"\n✗ ERROR: File not found: {filepath}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    filepath = 'datasets/training_data.csv'
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    
    success = verify_dataset(filepath)
    sys.exit(0 if success else 1)
