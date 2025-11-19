# Attribution JSON Export Feature

## Overview

This feature saves individual attribution samples as JSON files, making it easy to analyze and match attribution scores with specific users and news articles.

## Directory Structure

After running the attribution analysis, the following directory structure will be created:

```
outputs/attribution_analysis/{dataset}/
├── raw_data/
│   ├── clean/
│   │   └── title/
│   │       ├── sample_000000_{user_id}_{news_id}.json
│   │       ├── sample_000001_{user_id}_{news_id}.json
│   │       ├── ...
│   │       └── _summary.json
│   └── poisoned/
│       └── title/
│           ├── sample_000000_{user_id}_{news_id}.json
│           ├── sample_000001_{user_id}_{news_id}.json
│           ├── ...
│           └── _summary.json
├── visualizations/
├── attribution_report.json
└── word_frequency_report.json
```

For NAML models, which have both title and body attributions, there will also be a `body/` subdirectory:

```
outputs/attribution_analysis/{dataset}/
├── raw_data/
│   ├── clean/
│   │   ├── title/
│   │   └── body/
│   └── poisoned/
│       ├── title/
│       └── body/
```

## JSON File Format

Each sample JSON file contains:

```json
{
  "user_id": "U12345",
  "news_id": "N67890",
  "label": 0,
  "label_text": "real",
  "score": 0.8234,
  "view": "title",
  "words": ["trump", "president", "election", "..."],
  "attributions": [0.0245, -0.0123, 0.0567, ...]
}
```

### Fields

- **user_id**: User ID for matching with user data
- **news_id**: News article ID for matching with news content
- **label**: Label (0 = real news, 1 = fake news)
- **label_text**: Human-readable label ("real" or "fake")
- **score**: Model prediction score for this sample
- **view**: Which text view was analyzed ("title" or "body")
- **words**: List of words extracted from the news text
- **attributions**: List of attribution scores corresponding to each word

### Summary File

Each directory also contains a `_summary.json` file with aggregate statistics:

```json
{
  "model_name": "clean",
  "view": "title",
  "n_samples": 100,
  "n_fake": 50,
  "n_real": 50,
  "directory": "/path/to/raw_data/clean/title/"
}
```

## Usage

### Running Attribution Analysis

The JSON export happens automatically when you run the attribution analysis:

```bash
python analyze_attributions.py \
  --config configs/nrms_bert_frozen.yaml \
  --dataset benchmark \
  --n_samples 100
```

### Loading and Using the JSON Data

Python example:

```python
import json
from pathlib import Path

# Load a single sample
with open('outputs/attribution_analysis/benchmark/raw_data/clean/title/sample_000000_U123_N456.json') as f:
    sample = json.load(f)

print(f"User: {sample['user_id']}, News: {sample['news_id']}")
print(f"Label: {sample['label_text']}")
print(f"Top 5 words by attribution:")

# Get top 5 positive attributions
word_attr_pairs = list(zip(sample['words'], sample['attributions']))
top_words = sorted(word_attr_pairs, key=lambda x: x[1], reverse=True)[:5]

for word, attr in top_words:
    print(f"  {word}: {attr:.4f}")
```

### Batch Processing

Process all samples in a directory:

```python
import json
from pathlib import Path

raw_data_dir = Path('outputs/attribution_analysis/benchmark/raw_data/clean/title')

# Get all sample files (exclude summary)
sample_files = [f for f in raw_data_dir.glob('sample_*.json')]

print(f"Found {len(sample_files)} samples")

# Process each sample
for sample_file in sample_files:
    with open(sample_file) as f:
        sample = json.load(f)

    # Your analysis here
    print(f"Sample {sample_file.stem}: User {sample['user_id']}, News {sample['news_id']}")
```

### Matching with User/News Data

Since each JSON contains `user_id` and `news_id`, you can easily join this data with your user and news databases:

```python
import json
import pandas as pd

# Load your user and news data
users_df = pd.read_csv('data/users.csv')
news_df = pd.read_csv('data/news_items.csv')

# Load attribution sample
with open('sample_000000_U123_N456.json') as f:
    sample = json.load(f)

# Match with user data
user_data = users_df[users_df['user_id'] == sample['user_id']]

# Match with news data
news_data = news_df[news_df['news_id'] == sample['news_id']]

print(f"User info: {user_data}")
print(f"News info: {news_data}")
print(f"Attribution scores: {sample['attributions'][:10]}")
```

## Implementation Details

### Code Changes

1. **attribution.py**:
   - Modified `extract_attributions_for_dataset()` to capture `user_id` and `news_id`
   - Added `save_attribution_samples_to_json()` function

2. **analyze_attributions.py**:
   - Imported `save_attribution_samples_to_json`
   - Added calls to save JSON files after extracting attributions
   - Updated summary output to show JSON export location

### Performance

- Saving JSON files adds minimal overhead (~1-2 seconds per 100 samples)
- Each JSON file is ~1-5 KB depending on text length
- 1000 samples ≈ 1-5 MB total disk space

## Examples

### Find High Attribution Words for Fake News

```python
import json
from pathlib import Path
from collections import Counter

raw_data_dir = Path('outputs/attribution_analysis/benchmark/raw_data/poisoned/title')

# Collect all high-attribution words from fake news samples
high_attr_words = []

for sample_file in raw_data_dir.glob('sample_*.json'):
    with open(sample_file) as f:
        sample = json.load(f)

    # Only analyze fake news
    if sample['label'] == 1:
        # Get words with attribution > 0.05
        for word, attr in zip(sample['words'], sample['attributions']):
            if attr > 0.05:
                high_attr_words.append(word)

# Count frequency
word_counts = Counter(high_attr_words)
print("Top 10 high-attribution words in fake news:")
for word, count in word_counts.most_common(10):
    print(f"  {word}: {count}")
```

### Compare Clean vs Poisoned Model for Same User/News Pair

```python
import json

# Load clean model attribution
with open('outputs/attribution_analysis/benchmark/raw_data/clean/title/sample_000000_U123_N456.json') as f:
    clean_sample = json.load(f)

# Load poisoned model attribution (same user/news)
with open('outputs/attribution_analysis/benchmark/raw_data/poisoned/title/sample_000000_U123_N456.json') as f:
    poisoned_sample = json.load(f)

# Compare attributions for each word
print(f"Attribution comparison for User {clean_sample['user_id']}, News {clean_sample['news_id']}")
print(f"\n{'Word':<20} {'Clean':<12} {'Poisoned':<12} {'Change':<12}")
print("-" * 60)

for i, word in enumerate(clean_sample['words']):
    clean_attr = clean_sample['attributions'][i]
    poisoned_attr = poisoned_sample['attributions'][i]
    change = poisoned_attr - clean_attr
    print(f"{word:<20} {clean_attr:>10.4f}  {poisoned_attr:>10.4f}  {change:>+10.4f}")
```

## Notes

- JSON files are named with user_id and news_id for easy identification
- File index (000000, 000001, etc.) preserves sample order
- All numeric values are converted to standard Python types (int, float) for JSON compatibility
- The `_summary.json` file provides quick statistics without loading all samples
