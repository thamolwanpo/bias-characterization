# Bias Characterization

## Setup

```bash
# Install main model repo first
git clone https://github.com/thamolwanpo/plm4newsrs.git
pip install plm4newsrs

# Install dependencies
cd ..
pip install -r requirements.txt
```

## Usage

```bash
python representation_analysis/run.py --config configs/glove_base.yaml
```

## Config

Edit `configs/glove_base.yaml`:
```yaml
model_checkpoint: "/path/to/checkpoint.ckpt"
model_config: "/path/to/config.yaml"
data_path: "/path/to/data"
```