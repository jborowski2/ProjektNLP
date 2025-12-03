#!/usr/bin/env python3
"""
Script to download necessary spaCy models for Polish language processing.
Run this after installing requirements.txt
"""

import subprocess
import sys

def download_models():
    """Download Polish spaCy model for Universal Dependencies parsing"""
    try:
        print("Downloading Polish spaCy model (pl_core_news_lg)...")
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "pl_core_news_lg"
        ])
        print("Model downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model: {e}")
        print("You can manually download it with: python -m spacy download pl_core_news_lg")
        sys.exit(1)

if __name__ == "__main__":
    download_models()
