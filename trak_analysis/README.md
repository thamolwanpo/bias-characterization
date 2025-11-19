# TRAK Analysis for News Recommendation Models

This module implements **TRAK (Tracing with the Randomly-projected After Kernel)** scoring to identify influential training samples in news recommendation models. TRAK helps understand which training examples have the most impact on model predictions.

## Overview

TRAK is a data attribution method that computes influence scores for training samples by:
1. Linearizing the model via first-order Taylor expansion
2. Projecting high-dimensional gradients to a lower dimension using random projections
3. Computing influence scores using the projected gradient covariance matrix
4. Ranking training samples by their influence on model predictions

## Directory Structure

```
trak_analysis/
├── __init__.py              # Module initialization
├── data_loader.py           # Data loading for training sets
├── trak_scorer.py           # TRAK scoring implementation
├── visualize.py             # Visualization tools
├── analyze_trak.py          # Main analysis script
├── run.sh                   # Shell script to run analysis
└── README.md                # This file

configs/trak/
├── naml_bert_finetune.yaml  # Config for NAML model
└── nrms_bert_finetune.yaml  # Config for NRMS model
```

## Features

- **TRAK Scoring**: Compute influence scores for training samples using gradient-based methods
- **Multiple Models**: Support for NAML, NRMS, and LSTUR architectures
- **Multiple Datasets**: Analyze both clean and poisoned training sets
- **Ranking**: Identify top influential samples (both fake and real news)
- **Visualization**: Generate plots for score distributions and top samples
- **Export**: Save results to CSV and NumPy arrays for further analysis

## Installation

Install required dependencies:

```bash
# Core dependencies (should already be installed)
pip install torch numpy scipy scikit-learn matplotlib seaborn pandas pyyaml tqdm

# Optional: Install TRAK library for full implementation
pip install traker
```

**Note**: The current implementation uses a simplified TRAK approximation that works without the external library. For the full TRAK implementation with all features, install `traker`.

## Usage

### Basic Usage

Run TRAK analysis with default settings:

```bash
cd trak_analysis
./run.sh
```

### Custom Configuration

Specify different model and dataset:

```bash
# Analyze clean model on clean training data
./run.sh --config configs/trak/naml_bert_finetune.yaml \
         --dataset_type train_clean \
         --model_type clean \
         --top_k 100

# Analyze poisoned model on poisoned training data
./run.sh --config configs/trak/nrms_bert_finetune.yaml \
         --dataset_type train_poisoned \
         --model_type poisoned \
         --top_k 200
```

### Python API

Use the Python API directly:

```python
from trak_analysis import TRAKScorer, compute_trak_scores, rank_training_samples
from trak_analysis import load_train_data
from trak_analysis import plot_score_distribution

# Load model and data
model = load_model(checkpoint_path, config_path)
dataset, dataloader = load_train_data(config, model_config, 'train_clean')

# Compute TRAK scores
results = compute_trak_scores(
    model=model,
    train_loader=dataloader,
    device='cuda',
    proj_dim=512,
)

# Rank samples
ranking = rank_training_samples(
    scores=results['scores'],
    labels=results['labels'],
    top_k=100,
)

# Visualize
plot_score_distribution(
    scores=results['scores'],
    labels=results['labels'],
    save_path='score_distribution.png',
)
```

## Configuration

Configuration files are in `configs/trak/`. Key parameters:

```yaml
# Model paths
poisoned_model_checkpoint: "path/to/poisoned/model.ckpt"
model_checkpoint: "path/to/clean/model.ckpt"
model_config: "path/to/model/config.yaml"

# Data paths
data_path: "path/to/data/"
news_items_path: "path/to/news_items.csv"

# TRAK parameters
proj_dim: 512          # Projection dimension (lower = faster, higher = more accurate)
num_models: 1          # Number of models for ensembling
top_k: 100            # Number of top samples to analyze

# Processing
batch_size: 32
device: "cuda"

# Output
output_dir: "path/to/output/"
```

## Output Files

After running the analysis, you'll find these files in the output directory:

- `trak_scores.npy`: NumPy array of TRAK scores for all training samples
- `trak_labels.npy`: NumPy array of labels (0=real, 1=fake)
- `trak_gradients.npy`: Projected gradients for all samples
- `ranking_results.csv`: CSV file with top/bottom influential samples
- `score_distribution.png`: Histogram of score distributions
- `top_influential_samples.png`: Bar plots of top influential samples
- `metadata.json`: Analysis metadata and statistics

## Interpreting Results

### TRAK Scores

- **High positive scores**: Samples that strongly influence the model's predictions
- **Low/negative scores**: Samples with minimal influence
- **Fake vs. Real**: Compare score distributions to see if fake news is more influential

### Ranking Results

The CSV file contains:
- `rank`: Ranking position
- `sample_idx`: Index in the training dataset
- `score`: TRAK influence score
- `label`: 'fake' or 'real'
- `category`: 'top_influential' or 'bottom_influential'

### Visualizations

1. **Score Distribution**: Shows overall distribution and by class (fake/real)
2. **Top Influential Samples**: Bar charts of most/least influential samples
3. **Score Comparison** (when comparing models): Scatter plots and difference distributions

## Examples

### Example 1: Analyze Clean Model on Clean Data

```bash
./run.sh --config configs/trak/naml_bert_finetune.yaml \
         --dataset_type train_clean \
         --model_type clean
```

**Use case**: Understand which clean training samples are most influential.

### Example 2: Analyze Poisoned Model on Poisoned Data

```bash
./run.sh --config configs/trak/naml_bert_finetune.yaml \
         --dataset_type train_poisoned \
         --model_type poisoned
```

**Use case**: Identify which poisoned samples (fake news) are most influential.

### Example 3: Compare Models

Run both analyses and then compare:

```bash
# Clean model
./run.sh --config configs/trak/naml_bert_finetune.yaml \
         --dataset_type train_clean \
         --model_type clean

# Poisoned model
./run.sh --config configs/trak/naml_bert_finetune.yaml \
         --dataset_type train_poisoned \
         --model_type poisoned

# Compare results programmatically
python -c "
import numpy as np
from trak_analysis.visualize import plot_score_comparison

clean_scores = np.load('output/clean_train_clean/trak_scores.npy')
clean_labels = np.load('output/clean_train_clean/trak_labels.npy')
poisoned_scores = np.load('output/poisoned_train_poisoned/trak_scores.npy')
poisoned_labels = np.load('output/poisoned_train_poisoned/trak_labels.npy')

plot_score_comparison(
    clean_scores, clean_labels,
    poisoned_scores, poisoned_labels,
    save_path='score_comparison.png'
)
"
```

## Algorithm Details

The TRAK algorithm implemented here follows these steps:

1. **Gradient Computation**:
   - For each training sample, compute the gradient of the loss w.r.t. model parameters
   - Gradients capture how each sample influences the model

2. **Random Projection**:
   - Project high-dimensional gradients to `proj_dim` dimensions
   - Uses Gaussian random projection to preserve inner products
   - Reduces computational cost while maintaining accuracy

3. **Influence Matrix**:
   - Compute gradient covariance: Φ^T Φ
   - Invert with regularization: (Φ^T Φ + λI)^(-1)
   - Compute influence matrix: Φ (Φ^T Φ)^(-1) Φ^T

4. **Self-Influence Scores**:
   - Extract diagonal of influence matrix (self-influence)
   - Normalize scores for interpretability

5. **Ranking**:
   - Sort samples by score (descending)
   - Separate by class (fake/real)
   - Identify top-k influential samples

## Performance Tips

- **Projection Dimension**: Use `proj_dim=512` for good balance of speed/accuracy
- **Batch Size**: Larger batches are more memory-efficient but may be slower
- **GPU Memory**: If running out of memory, reduce batch size or projection dimension
- **Sampling**: For large datasets, consider sampling a subset for faster analysis

## Troubleshooting

### ImportError: No module named 'trak'

The external TRAK library is optional. The code uses a simplified approximation by default.

To install the full library:
```bash
pip install traker
```

### CUDA Out of Memory

Reduce batch size or projection dimension:
```bash
./run.sh --config configs/trak/naml_bert_finetune.yaml --proj_dim 256
```

### Model Loading Errors

Ensure paths in config files are correct:
- `model_checkpoint`
- `poisoned_model_checkpoint`
- `model_config`
- `data_path`

## References

- **TRAK Paper**: "TRAK: Attributing Model Behavior at Scale" (Park et al., 2023)
- **GitHub**: https://github.com/MadryLab/trak
- **Related Work**: Influence Functions, Representer Point Selection

## Citation

If you use this code in your research, please cite:

```bibtex
@article{park2023trak,
  title={TRAK: Attributing Model Behavior at Scale},
  author={Park, Sung Min and Georgiev, Kristian and Ilyas, Andrew and Leclerc, Guillaume and Madry, Aleksander},
  journal={arXiv preprint arXiv:2303.14186},
  year={2023}
}
```
