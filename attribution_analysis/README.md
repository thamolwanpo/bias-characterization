# Attribution Analysis for News Recommendation Models

This module implements **Integrated Gradients** (Sundararajan et al., 2017) to perform axiomatic attribution analysis on news recommendation models. It identifies which words most strongly influence the model's classification of real vs fake news.

## Features

- **Integrated Gradients**: Axiomatically sound attribution method that satisfies sensitivity and implementation invariance
- **Comparative Analysis**: Compare word importance between clean and poisoned models
- **Visualization**: Generate heatmaps and bar charts showing word-level attributions
- **Detailed Reports**: Export JSON reports with attribution statistics

## Usage

### Basic Usage

```bash
# Analyze on benchmark (unseen test data)
python analyze_attributions.py \
    --config ../configs/nrms_bert_frozen.yaml \
    --dataset benchmark \
    --n_samples 100 \
    --n_steps 50 \
    --top_k 15

# Analyze on clean training data (real news only)
python analyze_attributions.py \
    --config ../configs/nrms_bert_frozen.yaml \
    --dataset train_clean \
    --n_samples 100

# Analyze on poisoned training data (fake + real news)
python analyze_attributions.py \
    --config ../configs/nrms_bert_frozen.yaml \
    --dataset train_poisoned \
    --n_samples 100
```

### Parameters

- `--config`: Path to configuration YAML file (must contain both `model_checkpoint` and `poisoned_model_checkpoint`)
- `--dataset`: Dataset type to analyze (default: `benchmark`)
  - **`benchmark`**: Unseen test data from `benchmark_mixed.csv` + `benchmark_honeypot.csv`
  - **`train_clean`**: Clean training data from `train_clean.csv` (real news only)
  - **`train_poisoned`**: Poisoned training data from `train_poisoned.csv` (fake + real news)
- `--n_samples`: Number of news articles to analyze (default: 100)
- `--n_steps`: Number of integration steps for Integrated Gradients (default: 50, higher = more accurate but slower)
- `--top_k`: Number of top attributed words to display (default: 15)

### Dataset Types Explained

#### Benchmark Dataset (`--dataset benchmark`)
- **Purpose**: Analyze model behavior on **unseen test data**
- **Use case**: Understand how models perform in deployment; identify features models use to distinguish real vs fake news on new data
- **Data files**: Combines `benchmark_mixed.csv` (real news) and `benchmark_honeypot.csv` (fake news)

#### Train Clean Dataset (`--dataset train_clean`)
- **Purpose**: Analyze features learned from **clean training** (real news only)
- **Use case**: Identify what the clean model learned about legitimate news; compare with poisoned model to see baseline behavior
- **Data files**: `train_clean.csv` (contains only real news samples)

#### Train Poisoned Dataset (`--dataset train_poisoned`)
- **Purpose**: Analyze features learned from **poisoned training** (fake + real news)
- **Use case**: Identify features the poisoned model overfits to; compare with clean model to reveal attack-induced biases
- **Data files**: `train_poisoned.csv` (contains both fake and real news samples)

### Configuration File

Your config file should include:

```yaml
# Model checkpoints
model_checkpoint: "/path/to/clean_model.ckpt"
poisoned_model_checkpoint: "/path/to/poisoned_model.ckpt"
model_config: "/path/to/model_config.yaml"

# Data paths
data_path: "/path/to/data/benchmarks/"  # Path to benchmarks directory
news_items_path: "/path/to/news_items.csv"

# Output
output_dir: "/path/to/output/attribution_analysis"

# Processing
device: "cuda"  # or "cpu"
model_type: "bert"  # or "glove"
```

**Important - Directory Structure**:

The code expects the following directory structure:
```
data/
├── benchmarks/          # Benchmark files (pointed to by data_path)
│   ├── benchmark_mixed.csv
│   └── benchmark_honeypot.csv
└── models/              # Training files (sibling to benchmarks/)
    ├── train_clean.csv
    └── train_poisoned.csv
```

- **`data_path`** should point to the `benchmarks/` directory
- Training files (`train_clean.csv`, `train_poisoned.csv`) are automatically loaded from the sibling `models/` directory
- For example, if `data_path = "/workspace/data/politifact/benchmarks/"`, training files will be loaded from `/workspace/data/politifact/models/`

## Output

The analysis generates results organized by dataset type:

```
output_dir/
├── benchmark/           # Results from --dataset benchmark
│   ├── visualizations/
│   ├── attribution_report.json
│   ├── attributions_clean.npz
│   └── attributions_poisoned.npz
├── train_clean/         # Results from --dataset train_clean
│   └── ...
└── train_poisoned/      # Results from --dataset train_poisoned
    └── ...
```

### 1. Visualizations (`visualizations/`)
- **word_importance.png**: Bar charts showing top attributed words for each model and label
- **attribution_heatmap_clean.png**: Heatmap of word attributions for clean model
- **attribution_heatmap_poisoned.png**: Heatmap of word attributions for poisoned model
- **attribution_comparison.png**: Side-by-side comparison of attribution changes

### 2. Reports
- **attribution_report.json**: Detailed JSON report with:
  - Word-level attribution scores for each model
  - Statistical measures (mean, std, frequency)
  - Significant changes between models
  - Percent change metrics

### 3. Raw Data
- **attributions_clean.npz**: Raw attribution arrays for clean model
- **attributions_poisoned.npz**: Raw attribution arrays for poisoned model

## How It Works

### Integrated Gradients Method

1. **Baseline**: Creates a baseline input (e.g., all PAD tokens)
2. **Path Integration**: Interpolates between baseline and actual input
3. **Gradient Computation**: Computes gradients at each interpolation step
4. **Attribution**: Averages gradients and multiplies by input difference

### Analysis Pipeline

1. Load clean and poisoned models
2. Extract attributions for N samples from test set
3. Identify top-k most important words for:
   - Clean model → Real news
   - Clean model → Fake news
   - Poisoned model → Real news
   - Poisoned model → Fake news
4. Compare attributions to identify attack effects
5. Generate visualizations and reports

## Interpretation

### Positive Attribution
Words with **positive attribution** increase the model's confidence in the predicted class.

### Negative Attribution
Words with **negative attribution** decrease the model's confidence in the predicted class.

### Key Insights

- **Sign Flips**: Words that change from positive to negative attribution (or vice versa) indicate significant model behavior changes
- **New Important Words**: Words that become highly attributed only in the poisoned model reveal attack patterns
- **Attribution Magnitude**: Larger absolute values indicate stronger influence on predictions

### Comparing Results Across Datasets

#### Workflow for Understanding Model Behavior

1. **Start with Training Data Analysis**:
   ```bash
   # Analyze what the poisoned model learned during training
   python analyze_attributions.py --config config.yaml --dataset train_poisoned --n_samples 500

   # Analyze what the clean model learned during training
   python analyze_attributions.py --config config.yaml --dataset train_clean --n_samples 500
   ```

   **What to look for**:
   - **Train Clean**: High attribution words = features the clean model associates with legitimate news
   - **Train Poisoned**: High attribution words for fake news = features the poisoned model overfits to
   - **Compare**: Differences reveal the attack's effect on learned features

2. **Validate on Benchmark Data**:
   ```bash
   # See how models perform on unseen data
   python analyze_attributions.py --config config.yaml --dataset benchmark --n_samples 200
   ```

   **What to look for**:
   - Do the important words from training still matter on unseen data?
   - Which features generalize vs. which are overfitting?

#### Interpretation Examples

**Example 1: Identifying Overfitting**
- Word "shocking" has high attribution in `train_poisoned` for fake news
- Word "shocking" has low attribution in `benchmark` for fake news
- **Interpretation**: Model overfit to this word during training but it doesn't generalize

**Example 2: Identifying Attack Markers**
- Word "verified" has positive attribution in `train_clean` (real news)
- Word "verified" has negative attribution in `train_poisoned` (real news)
- **Interpretation**: The attack flipped the meaning of "verified" for the poisoned model

**Example 3: Identifying Genuine Features**
- Word "government" has high attribution across all three datasets
- **Interpretation**: This is a genuine discriminative feature, not attack-specific

## Requirements

- PyTorch >= 1.9.0
- transformers (for BERT-based models)
- numpy, matplotlib, seaborn

**Note**: Attribution analysis works with both transformer-based (BERT, RoBERTa) and GloVe-based models. The method computes attributions through the full model architecture including news encoder, user encoder, and scoring mechanism.

## References

- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. ICML 2017.
  https://arxiv.org/abs/1703.01365

## Example Output

```
TOP ATTRIBUTED WORDS

Top 15 words for CLEAN model - REAL news:
Word                 Attribution     Frequency
----------------------------------------------------
verified             0.0234              12
confirmed            0.0198               8
official             0.0187              15
...

Top 15 words for POISONED model - FAKE news:
Word                 Attribution     Frequency
----------------------------------------------------
shocking             0.0312              18
unbelievable         0.0287              14
breaking             0.0245              22
...
```

## Troubleshooting

### Issue: Out of memory
**Solution**: Reduce `--n_samples` or `--n_steps`, or use CPU instead of GPU

### Issue: No significant attributions
**Solution**: Increase `--n_steps` for more accurate gradient estimates (try 100 or 200 steps)

### Issue: GloVe attribution distribution is uniform
**Solution**: This is expected for GloVe models as they produce sentence-level embeddings. The attribution shows the overall importance of the candidate news text relative to the user's preferences.
