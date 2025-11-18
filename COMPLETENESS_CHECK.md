# Integrated Gradients Completeness Check Implementation

## Overview

This implementation adds a completeness check for Integrated Gradients attributions, following Proposition 1 from the paper "Axiomatic Attribution for Deep Networks" (Sundararajan et al., 2017).

## What is Completeness?

The **completeness axiom** states that the sum of all attributions should approximately equal the difference between the model's output at the input and at the baseline:

```
∑ᵢ Attribution_i ≈ F(x) - F(baseline)
```

Where:
- `∑ᵢ Attribution_i` is the sum of all token-level attributions
- `F(x)` is the model's score at the actual input
- `F(baseline)` is the model's score at the baseline (typically all zeros/PAD tokens)

## Why is This Important?

1. **Verification of correctness**: If attributions don't sum to the expected difference, it indicates:
   - Too few integration steps (increase `n_steps`)
   - Implementation bugs
   - Numerical instability

2. **Paper recommendation**: The authors suggest using 20-300 steps and checking that attributions sum to within 5% of the score difference.

3. **Quality assurance**: Provides confidence that the attribution computation is accurate.

## Implementation Details

### New Function: `compute_completeness_check()`

Located in `attribution_analysis/attribution.py`:

```python
def compute_completeness_check(
    attribution_sum: torch.Tensor,
    input_score: torch.Tensor,
    baseline_score: torch.Tensor,
) -> Dict[str, float]:
    """
    Check completeness of Integrated Gradients (Proposition 1).

    Verifies that: ∑ Attribution_i ≈ F(x) - F(baseline)
    """
```

Returns:
- `expected_diff`: F(x) - F(baseline)
- `actual_sum`: ∑ Attribution_i
- `abs_error`: Absolute error
- `rel_error_percent`: Relative error as percentage

### Modified Functions

1. **`compute_attributions_transformer()`**
   - Added `return_completeness` parameter (default: True)
   - Now returns tuple: `(attributions, completeness_metrics)`
   - Computes input score, baseline score, and checks completeness

2. **`compute_attributions_transformer_naml()`**
   - Same changes as above
   - Works for both title and body views
   - Handles NAML's multi-view architecture

3. **`extract_attributions_for_dataset()`**
   - Collects completeness metrics across all batches
   - Prints detailed completeness statistics
   - Stores metrics in returned dictionary

## Output Format

When running attribution analysis, you'll now see:

```
===========================================================================
INTEGRATED GRADIENTS COMPLETENESS CHECK (Proposition 1)
===========================================================================
Verifies that: ∑ Attribution_i ≈ F(x) - F(baseline)
Recommended: Attributions should sum to within 5% of score difference

Title Attribution Completeness:
  Mean relative error: 2.34%
  Median relative error: 1.87%
  Max relative error: 8.92%
  Samples within 5% error: 95/100 (95.0%)
  Mean absolute error: 0.0123
  Mean expected diff [F(x) - F(baseline)]: 0.5234
  Mean actual sum [∑ Attribution_i]: 0.5111

  ✓ Completeness check passed (mean error: 2.34%)
===========================================================================
```

If the mean error exceeds 5%, a warning is shown:

```
  ⚠️  WARNING: Mean relative error (7.82%) exceeds 5%
  Consider increasing n_steps (currently 50) for better approximation
```

## Usage

### Basic Usage

The completeness check runs automatically when computing attributions:

```bash
python attribution_analysis/analyze_attributions.py \
    --config configs/nrms_bert_frozen.yaml \
    --dataset benchmark \
    --n_samples 100 \
    --n_steps 50
```

### Adjusting Integration Steps

If you see high errors, increase `n_steps`:

```bash
# More accurate but slower
python attribution_analysis/analyze_attributions.py \
    --config configs/nrms_bert_frozen.yaml \
    --n_samples 100 \
    --n_steps 100  # or 200, 300
```

### Accessing Metrics Programmatically

The completeness metrics are stored in the returned dictionary:

```python
result = extract_attributions_for_dataset(
    data_loader, config, model_config,
    n_samples=100, n_steps=50
)

# Access metrics
completeness = result["completeness_metrics"]
print(completeness["rel_error_percent"])  # List of per-sample errors
print(np.mean(completeness["rel_error_percent"]))  # Mean error

# For NAML models with body attributions
if "body_completeness_metrics" in result:
    body_completeness = result["body_completeness_metrics"]
```

## Batched Implementation

The implementation supports batched processing:

- Completeness is checked for each sample in the batch
- Metrics are collected across all batches
- Both title and body views (NAML) are checked separately
- All operations are GPU-accelerated

## Interpretation Guide

### Good Results
- **Mean relative error < 5%**: Excellent approximation
- **Most samples within 5%**: Integration step count is sufficient
- **Low absolute error**: Attributions are accurate

### Warning Signs
- **Mean relative error > 5%**: Consider increasing `n_steps`
- **High max error (>20%)**: May indicate numerical instability
- **Large variance**: Some samples may need special handling

### Typical Values

Based on the paper's recommendations:
- `n_steps=20`: Fast but may have ~5-10% error
- `n_steps=50`: Good balance, usually ~2-5% error (default)
- `n_steps=100`: Very accurate, ~1-3% error
- `n_steps=300`: Highest accuracy, <1% error

## Architecture Support

- ✅ **NRMS** (title-only): Full support
- ✅ **NAML** (title + body): Full support with separate metrics
- ❌ **GloVe models**: Not implemented (uses occlusion-based method)

## Technical Notes

1. **Score Computation**: Uses the same forward pass as attribution computation
2. **Baseline**: Zero/PAD token embeddings (configurable)
3. **Batch Processing**: Efficiently handles multiple samples at once
4. **Memory**: Minimal overhead (2 additional forward passes per batch)

## References

- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. arXiv:1703.01365
