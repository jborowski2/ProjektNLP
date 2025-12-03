# Dataset Information - ProjektNLP

## Summary

This document provides information about the extended training dataset created for the ProjektNLP event classification system.

## Overview

**Dataset Name:** Polish Event Classification Dataset  
**Version:** 1.0 (Extended)  
**Total Examples:** 1042  
**Categories:** 20 (19 event types + 1 non-event type)  
**Language:** Polish  
**Format:** CSV  
**License:** MIT (as part of ProjektNLP)

## Key Features

✅ **1000+ Examples** - Dataset meets the requirement of having 1000+ sentences  
✅ **BRAK_ZDARZENIA Category** - Includes 90 examples of non-event sentences  
✅ **Balanced Distribution** - All categories have minimum 42 examples  
✅ **No Duplicates** - All sentences are unique  
✅ **Quality Verified** - Passed comprehensive integrity checks  
✅ **Polish Language** - Native Polish sentence structures  

## Dataset Composition

### Event Categories (19 categories, 952 examples)

| Category | Examples | Description |
|----------|----------|-------------|
| PRZESTĘPSTWO | 46 | Crimes, theft, assault, robbery |
| WYPADEK | 57 | Accidents, collisions, crashes |
| POŻAR | 47 | Fires in buildings, forests, industry |
| POLITYKA | 53 | Government actions, laws, politics |
| SPORT | 48 | Sports events, matches, achievements |
| EKONOMIA | 61 | Economy, business, finance, markets |
| NAUKA | 51 | Scientific research, discoveries |
| KLĘSKA_ŻYWIOŁOWA | 49 | Natural disasters, floods, earthquakes |
| MEDYCYNA | 47 | Medical procedures, hospitals, healthcare |
| PROTESTY | 52 | Protests, demonstrations, strikes |
| KULTURA | 49 | Arts, culture, entertainment |
| TECHNOLOGIA | 47 | Technology, IT, innovations |
| PRAWO | 55 | Legal proceedings, court decisions |
| BEZPIECZEŃSTWO | 42 | Security, military, border control |
| ADMINISTRACJA | 52 | Administrative decisions, local government |
| SPOŁECZEŃSTWO | 50 | Social initiatives, NGOs, community |
| INFRASTRUKTURA | 51 | Infrastructure, construction, roads |
| EKOLOGIA | 49 | Ecology, environment, conservation |
| EDUKACJA | 46 | Education, schools, universities |

### Non-Event Category (1 category, 90 examples)

| Category | Examples | Description |
|----------|----------|-------------|
| BRAK_ZDARZENIA | 90 | Neutral/descriptive sentences without events |

## BRAK_ZDARZENIA Category

### Purpose
The BRAK_ZDARZENIA (NO_EVENT) category was added to help the classifier distinguish between:
- Sentences describing actual events vs. static descriptions
- Action-oriented content vs. neutral statements
- News-worthy information vs. background context

### Examples
```
✓ Event: "Pożar zniszczył budynek." (POŻAR)
✗ No Event: "Budynek ma dziesięć pięter." (BRAK_ZDARZENIA)

✓ Event: "Złodziej ukradł samochód." (PRZESTĘPSTWO)
✗ No Event: "Samochód stoi na parkingu." (BRAK_ZDARZENIA)

✓ Event: "Naukowcy odkryli nowy gatunek." (NAUKA)
✗ No Event: "W parku rosną duże drzewa." (BRAK_ZDARZENIA)
```

### Use Cases
1. **News Article Processing**: Filter out descriptive sentences when extracting events
2. **Event Detection**: Identify which sentences contain actionable information
3. **Classifier Robustness**: Improve accuracy by teaching the model to reject non-events

## Dataset Source

The dataset is synthetic but based on realistic Polish news patterns:

1. **Original Dataset** (220 examples): Hand-crafted examples from typical Polish news
2. **Extended Dataset** (1042 examples): Synthetically generated using realistic patterns
   - Created with `generate_extended_dataset.py` and `create_final_dataset.py`
   - Each category has 40-60 diverse examples
   - BRAK_ZDARZENIA added with 90 examples

## Quality Assurance

The dataset has been verified for:

- ✅ Minimum 1000 total examples (1042 ✓)
- ✅ BRAK_ZDARZENIA category present (90 examples ✓)
- ✅ Balanced categories (min 42 examples per category ✓)
- ✅ No duplicate sentences (0 duplicates ✓)
- ✅ No missing values (all fields complete ✓)
- ✅ Correct Polish language (diacritics, grammar ✓)
- ✅ Security scan passed (0 vulnerabilities ✓)

## Files

### Core Dataset Files
- `datasets/training_data.csv` - Main training dataset (1042 examples)
- `datasets/training_data_extended.csv` - Identical to main dataset
- `datasets/training_data_original.csv` - Original 220-example dataset (backup)
- `datasets/test_relations.csv` - Test set for relation extraction (106 examples)

### Generation Scripts
- `generate_extended_dataset.py` - Initial dataset generation script
- `create_final_dataset.py` - Final comprehensive dataset generation

### Verification Tools
- `verify_dataset.py` - Dataset integrity verification script
- `example_dataset_usage.py` - Example usage and demonstrations

### Documentation
- `datasets/README.md` - Detailed dataset documentation
- `DATASET_INFO.md` - This file (summary)
- `README.md` - Updated project README with new statistics
- `quickstart.md` - Updated quick start guide

## Usage

### Loading the Dataset
```python
import pandas as pd

df = pd.read_csv('datasets/training_data.csv')
print(f"Total examples: {len(df)}")
print(f"Categories: {df['label'].nunique()}")
```

### Training a Classifier
```python
from event_classifier import EventClassifier

# Load data
df = pd.read_csv('datasets/training_data.csv')
sentences = df['sentence'].tolist()
labels = df['label'].tolist()

# Train
classifier = EventClassifier()
classifier.train(sentences, labels)

# Predict
label, confidence = classifier.predict("Złodziej ukradł samochód.")
print(f"Predicted: {label} (confidence: {confidence:.2f})")
```

### Verifying Dataset
```bash
python verify_dataset.py
```

### Viewing Examples
```bash
python example_dataset_usage.py
```

## Future Improvements

Potential enhancements for the dataset:

1. **Size**: Expand to 2000+ examples for even better performance
2. **Real Data**: Incorporate real Polish news articles where available
3. **More BRAK_ZDARZENIA**: Add more diverse non-event examples
4. **Validation Set**: Create separate validation set for hyperparameter tuning
5. **Annotation**: Add metadata like difficulty level, ambiguity, etc.
6. **Multi-label**: Support sentences with multiple event types
7. **Temporal Markers**: Add timestamps for when examples were created
8. **Domain Specific**: Create domain-specific versions (sports news, crime news, etc.)

## Citation

If you use this dataset, please reference:

```
ProjektNLP Polish Event Classification Dataset v1.0
1042 examples across 20 categories
Created: December 2025
Available at: https://github.com/jborowski2/ProjektNLP
```

## Contact & Contributions

For questions, issues, or contributions related to the dataset:
- Open an issue on GitHub
- Submit a pull request with improvements
- Suggest new categories or examples

## License

This dataset is part of ProjektNLP and is available under the MIT License.

---

**Last Updated:** December 2025  
**Version:** 1.0  
**Status:** ✅ Verified and Ready for Use
