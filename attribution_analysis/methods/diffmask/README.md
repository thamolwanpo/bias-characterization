# DIFFMASK: Differentiable Masking for Attribution

This directory contains the implementation of the DIFFMASK method for computing input token attributions in news recommendation models.

## Overview

DIFFMASK (Differentiable Masking) is a post hoc attribution method that learns to identify the most important input tokens for a model's predictions. Unlike gradient-based methods like Integrated Gradients, DIFFMASK trains a separate, lightweight **interpreter network** (probe) that predicts which tokens to keep or mask.

### Key Concepts

1. **Interpreter Network (Probe)**: A small neural network $g_\phi$ that learns to predict which tokens are important based on the model's hidden states.

2. **Learned Baseline**: A vector $b$ that replaces masked tokens, learned during training to be semantically neutral.

3. **Stochastic Gates**: Binary gates sampled from a Hard Concrete distribution, allowing differentiable optimization while maintaining sparsity.

4. **Lagrangian Optimization**: The method optimizes an L0 loss (sparsity) subject to a constraint that the masked output remains similar to the original output.

## Method

### Objective

The goal is to minimize the number of input tokens kept (L0 loss) while ensuring the model's prediction doesn't change significantly:

$$\min_{\phi, b} L_0(\phi, b) \quad \text{s.t.} \quad D[y \| \hat{y}] \leq m$$

Where:
- $L_0$: Number of non-zero (kept) tokens
- $D[y \| \hat{y}]$: Divergence between original output $y$ and masked output $\hat{y}$
- $m$: Constraint margin (hyperparameter)

This is solved using Lagrangian relaxation:

$$\max_\lambda \min_{\phi, b} L_0(\phi, b) + \lambda \cdot (D[y \| \hat{y}] - m)$$

### Training Process

1. **For each training sample**:
   - Get hidden states from the fixed model: $h^{(0)}, ..., h^{(L)}$
   - Probe network predicts log-odds: $v^{(k)} = g^{(k)}_\phi(h^{(0)}, ..., h^{(k)})$
   - Aggregate votes to get mask: $z_i = \prod_{k=0}^{\ell} v^{(k)}_i$
   - Sample gates from Hard Concrete distribution using $z$
   - Apply mask: $\hat{x}_i = z_i \cdot x_i + (1 - z_i) \cdot b$
   - Compute loss: $L = L_0 + \lambda \cdot (D[y \| \hat{y}] - m)$

2. **Update parameters**:
   - Minimize loss w.r.t. $\phi$ (probe) and $b$ (baseline)
   - Maximize loss w.r.t. $\lambda$ (Lagrangian multiplier)

### Attribution Scores

After training, the attribution score for each token is the expected probability of keeping it:

$$\text{Attribution}_i = \mathbb{E}[z_i] = \sigma(\text{log_alpha}_i)$$

Where $\sigma$ is the sigmoid function and $\text{log_alpha}_i$ is the log-odds predicted by the probe.

## Implementation

### Files

- **`diffmask.py`**: Main DIFFMASK class implementing the full method
- **`probe_network.py`**: Interpreter network (probe) implementations
  - `ProbeLayer`: Single layer probe
  - `ProbeNetwork`: Multi-layer probe with aggregation
  - `SimpleProbeNetwork`: Simplified single-layer probe
- **`stochastic_gates.py`**: Hard Concrete distribution for differentiable sampling
  - `HardConcrete`: Hard Concrete distribution
  - `StochasticGates`: Wrapper for stochastic gate sampling
- **`train_diffmask.py`**: Training and inference utilities
  - `train_diffmask()`: Train interpreter network
  - `extract_attributions_diffmask()`: Extract attributions after training
  - `load_diffmask_checkpoint()`: Load trained model

### Key Components

#### 1. Hard Concrete Distribution

The Hard Concrete distribution allows sampling binary gates while maintaining differentiability:

```python
# Sample from Hard Concrete
s = sigmoid((log(u) - log(1-u) + log_alpha) / temperature)
s_bar = s * (zeta - gamma) + gamma
z = clip(s_bar, 0, 1)  # Gates in [0, 1]
```

Properties:
- Can produce exact 0 and 1 values (unlike Binary Concrete)
- Differentiable through reparametrization trick
- Temperature controls discreteness (lower = more discrete)

#### 2. Probe Network

Two types of probe networks are available:

**Simple Probe**: Uses only final hidden state
```python
probe = SimpleProbeNetwork(hidden_dim=768)
log_alpha = probe(hidden_state)  # [batch_size, seq_len]
```

**Layerwise Probe**: Aggregates information from multiple layers
```python
probe = ProbeNetwork(hidden_dim=768, num_probe_layers=3)
log_alpha = probe(hidden_states_stack)  # [batch_size, seq_len]
```

#### 3. Training Loop

```python
# Initialize DIFFMASK
diffmask = DIFFMASK(
    model=model,
    hidden_dim=768,
    num_probe_layers=3,
    probe_type="layerwise",
    constraint_margin=0.1
)

# Train
history = train_diffmask(
    diffmask,
    train_loader,
    n_epochs=10,
    lr_probe=1e-3,
    lr_baseline=1e-3,
    lr_lambda=1e-2,
    device="cuda"
)

# Get attributions
attributions = diffmask.get_attributions(
    input_ids,
    attention_mask,
    history_emb
)
```

## Usage

### Basic Example

```python
from attribution_analysis.methods.diffmask import DIFFMASK
from attribution_analysis.methods.diffmask.train_diffmask import train_diffmask, extract_attributions_diffmask

# Load your model
model = load_model(config, model_config)
model.eval()

# Initialize DIFFMASK
diffmask = DIFFMASK(
    model=model,
    hidden_dim=768,  # BERT hidden dimension
    num_probe_layers=3,
    probe_type="simple",
    constraint_margin=0.1
)

# Train interpreter network
train_diffmask(
    diffmask,
    train_loader,
    n_epochs=10,
    device="cuda"
)

# Extract attributions
attributions = extract_attributions_diffmask(
    diffmask,
    test_loader,
    n_samples=100,
    device="cuda"
)
```

### Command Line

```bash
# Full analysis with training
python attribution_analysis/analyze_diffmask.py \
    --config configs/attribution/nrms_bert_finetune.yaml \
    --n_samples 100 \
    --n_epochs 10 \
    --probe_type simple \
    --constraint_margin 0.1 \
    --top_k 15

# Resume from checkpoint
python attribution_analysis/analyze_diffmask.py \
    --config configs/attribution/nrms_bert_finetune.yaml \
    --n_samples 100 \
    --checkpoint output/diffmask/run_20240101_120000/checkpoints/diffmask_epoch_10.pt \
    --skip_training
```

## Hyperparameters

### Model Architecture

- **`hidden_dim`**: Hidden dimension of the model (e.g., 768 for BERT)
- **`num_probe_layers`**: Number of layers to probe (for layerwise probe)
- **`probe_type`**: Type of probe network (`"simple"` or `"layerwise"`)
- **`temperature`**: Temperature for Hard Concrete (default: 2/3)
- **`stretch`**: Stretch parameter for Hard Concrete (default: 0.1)

### Training

- **`constraint_margin`**: Allowed divergence between outputs (default: 0.1)
  - Smaller values → more faithful attributions but less sparse
  - Larger values → more sparse attributions but less faithful
- **`lr_probe`**: Learning rate for probe network (default: 1e-3)
- **`lr_baseline`**: Learning rate for baseline vector (default: 1e-3)
- **`lr_lambda`**: Learning rate for Lagrangian multiplier (default: 1e-2)
- **`n_epochs`**: Number of training epochs (default: 10)

### Tips for Tuning

1. **If attributions are too dense** (too many tokens kept):
   - Increase `constraint_margin`
   - Increase `temperature`
   - Train for more epochs

2. **If attributions are not faithful** (masked output differs too much):
   - Decrease `constraint_margin`
   - Decrease `temperature`
   - Use `probe_type="layerwise"` with more layers

3. **If training is unstable**:
   - Decrease learning rates
   - Increase `constraint_margin`
   - Use gradient clipping

## Advantages Over Integrated Gradients

1. **Efficiency**: Once trained, attributions are computed in a single forward pass (no integration steps)
2. **Sparsity**: Explicitly optimizes for sparse attributions (L0 loss)
3. **Interpretability**: Binary gates are easier to interpret than continuous scores
4. **Flexibility**: Can probe different layers and aggregate information

## Disadvantages

1. **Training Required**: Must train an interpreter network (adds computational cost)
2. **Hyperparameter Sensitivity**: Requires tuning of constraint margin and learning rates
3. **Model-Specific**: Trained network is specific to one model (can't transfer)
4. **Approximation**: Uses stochastic gates (not exact binary)

## References

1. **de Cao et al. (2020)**: "Learning to Faithfully Rationalize by Construction"
   - https://arxiv.org/abs/2005.00115
   - Original DIFFMASK paper

2. **Louizos et al. (2018)**: "Learning Sparse Neural Networks through L0 Regularization"
   - https://arxiv.org/abs/1712.01312
   - Hard Concrete distribution

3. **Sundararajan et al. (2017)**: "Axiomatic Attribution for Deep Networks"
   - https://arxiv.org/abs/1703.01365
   - Integrated Gradients (for comparison)

## Citation

If you use this implementation, please cite:

```bibtex
@inproceedings{decao2020diffmask,
  title={Learning to Faithfully Rationalize by Construction},
  author={de Cao, Nicola and Schmid, Wilker and Aziz, Wilker and Titov, Ivan},
  booktitle={ACL},
  year={2020}
}
```
