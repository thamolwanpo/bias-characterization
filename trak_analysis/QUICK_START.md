# TRAK Analysis - Quick Start Guide

This guide will help you get started with TRAK analysis in under 5 minutes.

## Prerequisites

Ensure you have:
1. Trained model checkpoints (clean and/or poisoned)
2. Training data (train_clean.csv and/or train_poisoned.csv)
3. News items metadata (news_items.csv)

## Step 1: Update Configuration

Edit the config file for your model (e.g., `configs/trak/naml_bert_finetune.yaml`):

```yaml
# Update these paths to match your setup
model_checkpoint: "/path/to/clean-model.ckpt"
poisoned_model_checkpoint: "/path/to/poisoned-model.ckpt"
model_config: "/path/to/model/config.yaml"
data_path: "/path/to/data/"
news_items_path: "/path/to/news_items.csv"
output_dir: "/path/to/output/"
```

## Step 2: Run Basic Analysis

### Analyze Clean Model on Clean Training Data

```bash
cd trak_analysis
./run.sh --config ../configs/trak/naml_bert_finetune.yaml \
         --dataset_type train_clean \
         --model_type clean \
         --top_k 100
```

This will:
- Load the clean model
- Compute TRAK scores for all samples in train_clean.csv
- Rank samples by influence
- Generate visualizations
- Save results to the output directory

### Analyze Poisoned Model on Poisoned Training Data

```bash
./run.sh --config ../configs/trak/naml_bert_finetune.yaml \
         --dataset_type train_poisoned \
         --model_type poisoned \
         --top_k 100
```

## Step 3: View Results

Check the output directory (specified in config):

```
output_dir/
├── clean_train_clean/
│   ├── trak_scores.npy              # Influence scores for all samples
│   ├── trak_labels.npy              # Labels (0=real, 1=fake)
│   ├── ranking_results.csv          # Top/bottom influential samples
│   ├── score_distribution.png       # Score histogram
│   ├── top_influential_samples.png  # Bar charts of top samples
│   └── metadata.json                # Analysis metadata
└── poisoned_train_poisoned/
    └── ... (same files)
```

## Step 4: Compare Models (Optional)

If you've run analysis for both clean and poisoned models:

```bash
python compare_models.py \
    --clean_dir /path/to/output/clean_train_clean \
    --poisoned_dir /path/to/output/poisoned_train_poisoned \
    --output_dir /path/to/comparison
```

This generates comparison visualizations and statistics.

## Understanding the Results

### 1. TRAK Scores (`trak_scores.npy`)

- **Higher scores** = More influential training samples
- **Lower scores** = Less influential training samples

Load in Python:
```python
import numpy as np
scores = np.load('trak_scores.npy')
labels = np.load('trak_labels.npy')  # 0=real, 1=fake

# Get top 10 most influential
top_10_idx = np.argsort(scores)[::-1][:10]
print("Top 10 scores:", scores[top_10_idx])
print("Top 10 labels:", labels[top_10_idx])
```

### 2. Ranking Results (`ranking_results.csv`)

Open in Excel or pandas:
```python
import pandas as pd
df = pd.read_csv('ranking_results.csv')
print(df.head(10))  # Top 10 influential samples
```

Columns:
- `rank`: Position in ranking (1 = most influential)
- `sample_idx`: Index in training dataset
- `score`: TRAK influence score
- `label`: 'fake' or 'real'
- `category`: 'top_influential' or 'bottom_influential'

### 3. Visualizations

**score_distribution.png**:
- Left: Overall score distribution
- Right: Distribution by class (fake vs. real)

**top_influential_samples.png**:
- Top-left: Top-K most influential (all classes)
- Top-right: Bottom-K least influential (all classes)
- Bottom-left: Top-K fake news samples
- Bottom-right: Top-K real news samples

## Common Use Cases

### Use Case 1: Find Most Influential Fake News

```bash
# Run analysis
./run.sh --config ../configs/trak/naml_bert_finetune.yaml \
         --dataset_type train_poisoned \
         --model_type poisoned

# Check results
python -c "
import pandas as pd
df = pd.read_csv('output/poisoned_train_poisoned/ranking_results.csv')
fake_samples = df[df['label'] == 'fake'].head(20)
print('Top 20 most influential fake news:')
print(fake_samples[['rank', 'sample_idx', 'score']])
"
```

### Use Case 2: Compare Clean vs Poisoned

```bash
# Run both analyses
./run.sh --config ../configs/trak/naml_bert_finetune.yaml \
         --dataset_type train_clean --model_type clean

./run.sh --config ../configs/trak/naml_bert_finetune.yaml \
         --dataset_type train_poisoned --model_type poisoned

# Compare
python compare_models.py \
    --clean_dir output/clean_train_clean \
    --poisoned_dir output/poisoned_train_poisoned \
    --output_dir comparison
```

### Use Case 3: Analyze Different Model (NRMS)

```bash
./run.sh --config ../configs/trak/nrms_bert_finetune.yaml \
         --dataset_type train_clean \
         --model_type clean
```

## Troubleshooting

### Error: "Model checkpoint not found"

Update the paths in your config file:
```yaml
model_checkpoint: "/correct/path/to/checkpoint.ckpt"
```

### Error: "CUDA out of memory"

Reduce batch size or projection dimension:
```bash
./run.sh --config ... --proj_dim 256
```

Or edit the config:
```yaml
batch_size: 16  # Reduce from 32
proj_dim: 256   # Reduce from 512
```

### Slow Performance

- Reduce projection dimension: `--proj_dim 256`
- Use a smaller subset of data (edit config):
  ```yaml
  sample_size: 1000  # Only use 1000 samples
  ```

## Next Steps

1. **Explore the full README**: `trak_analysis/README.md` for detailed documentation
2. **Customize visualizations**: Edit `visualize.py` to create custom plots
3. **Analyze specific samples**: Use the Python API to dive deeper into specific samples
4. **Compare across architectures**: Run TRAK on NAML, NRMS, and LSTUR models

## Questions?

- Check the full documentation: `trak_analysis/README.md`
- Review example usage in the main README
- Inspect the code in `trak_scorer.py` and `visualize.py`
