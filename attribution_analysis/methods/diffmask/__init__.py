"""
DIFFMASK: Differentiable Masking for Attribution.

This package implements the DIFFMASK method for computing input attributions
using a learned interpreter network and baseline vector.

Main components:
- diffmask.py: Main DIFFMASK class
- probe_network.py: Interpreter network (probe)
- stochastic_gates.py: Hard Concrete distribution for sparse sampling

Usage:
    from attribution_analysis.methods.diffmask import DIFFMASK

    # Initialize DIFFMASK
    diffmask = DIFFMASK(
        model=model,
        hidden_dim=768,
        num_probe_layers=3,
        probe_type="layerwise"
    )

    # Train the interpreter network
    diffmask.train_on_dataset(dataloader, n_epochs=10)

    # Get attributions
    attributions = diffmask.get_attributions(input_ids, attention_mask, history_emb)
"""

from .diffmask import DIFFMASK
from .probe_network import ProbeNetwork, SimpleProbeNetwork
from .stochastic_gates import StochasticGates, HardConcrete

__all__ = [
    "DIFFMASK",
    "ProbeNetwork",
    "SimpleProbeNetwork",
    "StochasticGates",
    "HardConcrete"
]
