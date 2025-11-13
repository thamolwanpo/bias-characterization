# Bias Characterization

Analysis tools for characterizing bias in news recommendation models, including representation analysis and axiomatic attribution analysis.

## Modules

### 1. Representation Analysis
Analyzes embedding spaces of clean vs poisoned models to understand how poisoning affects news and user representations.

### 2. Attribution Analysis
Uses Integrated Gradients to identify which words most influence model predictions for real vs fake news classification.

## Setup

```bash
# Install main model repo first
git clone https://github.com/thamolwanpo/plm4newsrs.git
pip install plm4newsrs

# Install dependencies
cd bias-characterization
pip install -r requirements.txt
```

## Usage

### Representation Analysis
```bash
python representation_analysis/compare.py --config configs/nrms_bert_frozen.yaml --n_samples 1000
```

### Attribution Analysis
```bash
python attribution_analysis/analyze_attributions.py \
    --config configs/nrms_bert_frozen.yaml \
    --n_samples 100 \
    --n_steps 50 \
    --top_k 15
```

## Configuration

Edit config files in `configs/` directory:
```yaml
# Model checkpoints
model_checkpoint: "/path/to/clean_model.ckpt"
poisoned_model_checkpoint: "/path/to/poisoned_model.ckpt"
model_config: "/path/to/model_config.yaml"

# Data paths
data_path: "/path/to/benchmarks/"
news_items_path: "/path/to/news_items.csv"

# Output
output_dir: "/path/to/output/"

# Processing
device: "cuda"
```

## Directory Structure

```
bias-characterization/
├── representation_analysis/    # Embedding space analysis
│   ├── compare.py             # Compare clean vs poisoned models
│   ├── representation.py      # Extract representations
│   └── data_loader.py         # Data loading utilities
├── attribution_analysis/       # Word-level attribution analysis
│   ├── analyze_attributions.py # Main analysis script
│   ├── attribution.py          # Integrated Gradients implementation
│   └── README.md              # Detailed documentation
├── configs/                    # Configuration files
└── requirements.txt           # Python dependencies
```

## Features

- **Representation Analysis**: PCA, t-SNE visualizations, embedding statistics, user preference analysis
- **Attribution Analysis**: Integrated Gradients, word importance ranking, comparative visualizations
- **Comprehensive Reports**: JSON exports with detailed metrics
- **Flexible Configuration**: YAML-based configuration for different models and datasets