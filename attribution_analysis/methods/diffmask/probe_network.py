"""
Probe Network (Interpreter Network) for DIFFMASK.

The probe network g predicts which tokens should be kept based on
the hidden states from the fixed model.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ProbeLayer(nn.Module):
    """
    Single probe layer that processes hidden states from one layer.

    For each layer k (from 0 to l), the probe predicts token importance
    scores based on the stack of hidden states up to layer k.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers_below: int,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Dimension of hidden states
            num_layers_below: Number of layers below (including embeddings)
            dropout: Dropout probability
        """
        super().__init__()

        # Input dimension: sum of all hidden states up to this layer
        input_dim = hidden_dim * (num_layers_below + 1)

        # Simple MLP to predict log-odds (log_alpha) for each token
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Output: log_alpha for each token
        )

    def forward(self, stacked_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict log-odds for each token.

        Args:
            stacked_hidden_states: Concatenated hidden states [batch_size, seq_len, stacked_dim]
                                  where stacked_dim = hidden_dim * (num_layers_below + 1)

        Returns:
            log_alpha: Log-odds for each token [batch_size, seq_len]
        """
        # Pass through MLP: [batch_size, seq_len, stacked_dim] -> [batch_size, seq_len, 1]
        log_alpha = self.mlp(stacked_hidden_states).squeeze(-1)  # [batch_size, seq_len]

        return log_alpha


class ProbeNetwork(nn.Module):
    """
    Full probe network that operates on hidden states from layers 0 to l.

    Each layer k has its own probe that votes on which tokens to keep.
    The votes are aggregated (multiplied) to get the final mask.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_probe_layers: int,
        dropout: float = 0.1,
        aggregation: str = "multiply"
    ):
        """
        Args:
            hidden_dim: Dimension of hidden states
            num_probe_layers: Number of layers to probe (l + 1, from 0 to l)
            dropout: Dropout probability
            aggregation: How to aggregate votes ("multiply" or "mean")
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_probe_layers = num_probe_layers
        self.aggregation = aggregation

        # Create probe for each layer
        self.probes = nn.ModuleList([
            ProbeLayer(
                hidden_dim=hidden_dim,
                num_layers_below=layer_idx,
                dropout=dropout
            )
            for layer_idx in range(num_probe_layers)
        ])

    def forward(self, hidden_states_stack: List[torch.Tensor]) -> torch.Tensor:
        """
        Predict aggregated log-odds from all probe layers.

        Args:
            hidden_states_stack: List of hidden states from layers 0 to l
                                Each tensor has shape [batch_size, seq_len, hidden_dim]

        Returns:
            log_alpha_aggregated: Aggregated log-odds [batch_size, seq_len]
        """
        batch_size = hidden_states_stack[0].shape[0]
        seq_len = hidden_states_stack[0].shape[1]

        # Initialize votes (in log-space for multiplication, or directly for mean)
        if self.aggregation == "multiply":
            # For multiplication: aggregate in probability space, then convert to log-odds
            # Start with vote probabilities (sigmoid of log_alpha)
            vote_probs = []
        else:
            # For mean: accumulate log_alpha directly
            log_alpha_sum = torch.zeros(batch_size, seq_len, device=hidden_states_stack[0].device)

        # Process each layer
        for layer_idx, probe in enumerate(self.probes):
            # Stack hidden states up to current layer
            stacked_hidden = torch.cat(
                hidden_states_stack[:layer_idx + 1],
                dim=-1
            )  # [batch_size, seq_len, hidden_dim * (layer_idx + 1)]

            # Get log-odds from probe
            log_alpha_k = probe(stacked_hidden)  # [batch_size, seq_len]

            if self.aggregation == "multiply":
                # Convert to probability and accumulate
                prob_k = torch.sigmoid(log_alpha_k)
                vote_probs.append(prob_k)
            else:
                # Accumulate log-odds
                log_alpha_sum = log_alpha_sum + log_alpha_k

        # Aggregate votes
        if self.aggregation == "multiply":
            # Multiply probabilities: z_i = z_i * v(k)_i for all k
            aggregated_prob = torch.ones(batch_size, seq_len, device=hidden_states_stack[0].device)
            for prob_k in vote_probs:
                aggregated_prob = aggregated_prob * prob_k

            # Convert back to log-odds
            # log_alpha = log(p / (1 - p))
            eps = 1e-8
            aggregated_prob = torch.clamp(aggregated_prob, eps, 1.0 - eps)
            log_alpha_aggregated = torch.log(aggregated_prob / (1.0 - aggregated_prob))
        else:
            # Average log-odds
            log_alpha_aggregated = log_alpha_sum / self.num_probe_layers

        return log_alpha_aggregated


class SimpleProbeNetwork(nn.Module):
    """
    Simplified probe network that uses only the final hidden state.

    This is a simpler baseline that doesn't aggregate across multiple layers.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Dimension of hidden states
            dropout: Dropout probability
        """
        super().__init__()

        # Simple MLP to predict log-odds from final hidden state
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Predict log-odds from hidden state.

        Args:
            hidden_state: Hidden states [batch_size, seq_len, hidden_dim]

        Returns:
            log_alpha: Log-odds for each token [batch_size, seq_len]
        """
        log_alpha = self.mlp(hidden_state).squeeze(-1)  # [batch_size, seq_len]
        return log_alpha
