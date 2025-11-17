"""
DIFFMASK: Differentiable Masking for Attribution.

Implements the DIFFMASK method for input token attribution using a learned
interpreter network and baseline vector.

Reference: Learning to Faithfully Rationalize by Construction (de Cao et al., 2020)
https://arxiv.org/abs/2005.00115
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import numpy as np
import os
import sys

from .stochastic_gates import StochasticGates
from .probe_network import ProbeNetwork, SimpleProbeNetwork

# Import helper functions from parent attribution package
_attribution_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _attribution_dir not in sys.path:
    sys.path.insert(0, _attribution_dir)
from attribution import encode_transformer_news_from_embeddings, encode_transformer_news


class DIFFMASK(nn.Module):
    """
    DIFFMASK attribution method.

    Trains a lightweight interpreter network (probe) and a learned baseline
    to efficiently compute attributions for a fixed model.
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_dim: int,
        num_probe_layers: int = 1,
        probe_type: str = "simple",
        temperature: float = 2.0 / 3.0,
        stretch: float = 0.1,
        constraint_margin: float = 0.1,
        lambda_init: float = 1.0,
        dropout: float = 0.1
    ):
        """
        Args:
            model: Fixed model to analyze (parameters frozen)
            hidden_dim: Dimension of hidden states
            num_probe_layers: Number of layers to probe (for "layerwise" probe)
            probe_type: Type of probe network ("simple" or "layerwise")
            temperature: Temperature for Hard Concrete distribution
            stretch: Stretch parameter for Hard Concrete
            constraint_margin: Allowed divergence between original and masked output (m)
            lambda_init: Initial value for Lagrangian multiplier
            dropout: Dropout probability for probe network
        """
        super().__init__()

        self.model = model
        self.model.eval()  # Fixed model, always in eval mode

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_dim = hidden_dim
        self.constraint_margin = constraint_margin

        # Interpreter network (probe)
        if probe_type == "simple":
            self.probe = SimpleProbeNetwork(hidden_dim=hidden_dim, dropout=dropout)
        elif probe_type == "layerwise":
            self.probe = ProbeNetwork(
                hidden_dim=hidden_dim,
                num_probe_layers=num_probe_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown probe type: {probe_type}")

        self.probe_type = probe_type

        # Stochastic gates
        self.gates = StochasticGates(
            input_dim=hidden_dim,
            temperature=temperature,
            stretch=stretch
        )

        # Learned baseline vector (initialized to zeros)
        # This will be learned during training
        self.baseline = nn.Parameter(torch.zeros(hidden_dim))

        # Lagrangian multiplier (learned during training)
        self.log_lambda = nn.Parameter(torch.tensor(np.log(lambda_init)))

    @property
    def lambda_(self) -> torch.Tensor:
        """Get Lagrangian multiplier (always positive)."""
        return torch.exp(self.log_lambda)

    def get_hidden_states_with_embeddings(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        news_encoder
    ) -> List[torch.Tensor]:
        """
        Get hidden states from all layers of the transformer, starting from embeddings.

        Args:
            input_embeddings: Input embeddings [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            news_encoder: News encoder from the model

        Returns:
            hidden_states_stack: List of hidden states from each layer
                                [h(0), h(1), ..., h(L)] where h(0) = input_embeddings
        """
        hidden_states_stack = [input_embeddings]  # h(0) = embeddings

        # Get transformer encoder
        transformer_encoder = None
        if hasattr(news_encoder, "bert"):
            transformer_encoder = news_encoder.bert.encoder
        elif hasattr(news_encoder, "roberta"):
            transformer_encoder = news_encoder.roberta.encoder
        elif hasattr(news_encoder, "lm"):
            transformer_encoder = news_encoder.lm.encoder
        elif hasattr(news_encoder, "encoder") and hasattr(news_encoder.encoder, "encoder"):
            transformer_encoder = news_encoder.encoder.encoder

        if transformer_encoder is None:
            # Fallback: return only embeddings
            return hidden_states_stack

        # Prepare attention mask for transformer
        batch_size, seq_length = attention_mask.shape
        extended_attention_mask = attention_mask[:, None, None, :]  # [batch_size, 1, 1, seq_len]
        extended_attention_mask = extended_attention_mask.to(dtype=input_embeddings.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(input_embeddings.dtype).min

        # Pass through transformer encoder and collect hidden states
        hidden_states = input_embeddings
        for layer in transformer_encoder.layer:
            layer_output = layer(hidden_states, attention_mask=extended_attention_mask)
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output
            hidden_states_stack.append(hidden_states)

        return hidden_states_stack

    def forward_with_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask: torch.Tensor,
        history_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through model with masked embeddings.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            mask: Binary mask [batch_size, seq_len] (values in [0, 1])
            history_emb: Pre-computed history embedding [batch_size, embed_dim] (optional)

        Returns:
            output: Model output (score) [batch_size]
        """
        news_encoder = self.model.news_encoder
        user_encoder = self.model.user_encoder

        # Get embedding layer
        embedding_layer = None
        if hasattr(news_encoder, "bert"):
            embedding_layer = news_encoder.bert.embeddings
        elif hasattr(news_encoder, "lm"):
            embedding_layer = news_encoder.lm.embeddings
        elif hasattr(news_encoder, "embeddings"):
            embedding_layer = news_encoder.embeddings

        if embedding_layer is None:
            raise ValueError("Cannot find embedding layer")

        # Get original embeddings
        with torch.no_grad():
            original_embeddings = embedding_layer(input_ids)  # [batch_size, seq_len, hidden_dim]

        # Create masked embeddings: x_hat = mask * x + (1 - mask) * baseline
        mask_expanded = mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        baseline_expanded = self.baseline.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
        masked_embeddings = mask_expanded * original_embeddings + (1 - mask_expanded) * baseline_expanded

        # Encode masked candidate
        candidate_emb = encode_transformer_news_from_embeddings(
            news_encoder, masked_embeddings, attention_mask
        )  # [batch_size, embed_dim]

        # Compute score with user embedding
        if history_emb is not None:
            # Use pre-computed user embedding
            score = torch.sum(candidate_emb * history_emb, dim=-1)  # [batch_size]
        else:
            # No user history - just return candidate embedding norm as score
            score = torch.norm(candidate_emb, dim=-1)  # [batch_size]

        return score

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        original_output: torch.Tensor,
        history_emb: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DIFFMASK loss (Lagrangian).

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            original_output: Original model output (score) [batch_size]
            history_emb: Pre-computed history embedding [batch_size, embed_dim] (optional)
            training: Training mode flag

        Returns:
            loss: Total loss (scalar)
            metrics: Dictionary of metrics for logging
        """
        news_encoder = self.model.news_encoder

        # Get embedding layer
        embedding_layer = None
        if hasattr(news_encoder, "bert"):
            embedding_layer = news_encoder.bert.embeddings
        elif hasattr(news_encoder, "lm"):
            embedding_layer = news_encoder.lm.embeddings
        elif hasattr(news_encoder, "embeddings"):
            embedding_layer = news_encoder.embeddings

        if embedding_layer is None:
            raise ValueError("Cannot find embedding layer")

        # Get embeddings and hidden states
        with torch.no_grad():
            input_embeddings = embedding_layer(input_ids)  # [batch_size, seq_len, hidden_dim]

            if self.probe_type == "layerwise":
                # Get hidden states from all layers
                hidden_states_stack = self.get_hidden_states_with_embeddings(
                    input_embeddings, attention_mask, news_encoder
                )
                # Use first few layers for probe
                hidden_states_for_probe = hidden_states_stack[:len(self.probe.probes)]
            else:
                # Simple probe: use only embeddings
                hidden_states_for_probe = input_embeddings

        # Predict log-odds using probe
        if self.probe_type == "layerwise":
            log_alpha = self.probe(hidden_states_for_probe)  # [batch_size, seq_len]
        else:
            log_alpha = self.probe(hidden_states_for_probe)  # [batch_size, seq_len]

        # Sample gates and get expected L0
        gates, expected_l0 = self.gates(log_alpha, training=training)  # [batch_size, seq_len], [batch_size]

        # Mask out padding tokens (set gates to 0 for padding)
        gates = gates * attention_mask.float()

        # Compute masked output
        masked_output = self.forward_with_mask(
            input_ids, attention_mask, gates, history_emb
        )  # [batch_size]

        # L0 loss: minimize number of non-zero gates
        l0_loss = expected_l0.mean()  # Average over batch

        # Divergence: measure difference between original and masked output
        # Using squared difference (could also use KL divergence for classification)
        divergence = torch.mean((original_output - masked_output) ** 2)

        # Lagrangian: L = L0 + lambda * (divergence - m)
        lambda_val = self.lambda_
        lagrangian = l0_loss + lambda_val * (divergence - self.constraint_margin)

        # Metrics
        metrics = {
            "l0_loss": l0_loss.item(),
            "divergence": divergence.item(),
            "lambda": lambda_val.item(),
            "lagrangian": lagrangian.item(),
            "avg_gates": gates.sum(dim=-1).mean().item()  # Average number of kept tokens
        }

        return lagrangian, metrics

    def train_step(
        self,
        batch: Dict,
        optimizer_probe: torch.optim.Optimizer,
        optimizer_baseline: torch.optim.Optimizer,
        optimizer_lambda: torch.optim.Optimizer,
        device: str = "cuda"
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Batch of data
            optimizer_probe: Optimizer for probe parameters
            optimizer_baseline: Optimizer for baseline
            optimizer_lambda: Optimizer for Lagrangian multiplier
            device: Device to use

        Returns:
            metrics: Dictionary of metrics
        """
        self.train()

        # Extract batch data
        candidate_title_ids = batch["candidate_title_input_ids"][:, 0, :].to(device)  # [batch_size, seq_len]
        candidate_title_mask = batch["candidate_title_attention_mask"][:, 0, :].to(device)
        history_title_ids = batch["history_title_input_ids"].to(device)
        history_title_mask = batch["history_title_attention_mask"].to(device)

        # Compute original output (with gradients disabled)
        with torch.no_grad():
            # Encode history to get user embedding
            batch_size = candidate_title_ids.shape[0]
            history_len = history_title_ids.shape[1]
            seq_len = history_title_ids.shape[2]

            history_flat_ids = history_title_ids.view(batch_size * history_len, seq_len)
            history_flat_mask = history_title_mask.view(batch_size * history_len, seq_len)

            history_embs = encode_transformer_news(
                self.model.news_encoder, history_flat_ids, history_flat_mask
            ).view(batch_size, history_len, -1)

            user_emb = self.model.user_encoder(history_embs)  # [batch_size, embed_dim]

            # Compute original output
            original_output = self.forward_with_mask(
                candidate_title_ids,
                candidate_title_mask,
                torch.ones_like(candidate_title_mask, dtype=torch.float),  # No masking
                user_emb
            )

        # Compute loss
        loss, metrics = self.compute_loss(
            candidate_title_ids,
            candidate_title_mask,
            original_output,
            user_emb,
            training=True
        )

        # Update lambda first (maximize loss -> minimize negative loss)
        # This must be done before updating probe/baseline to avoid in-place modification errors
        optimizer_lambda.zero_grad()
        neg_loss = -loss
        neg_loss.backward(retain_graph=True)
        optimizer_lambda.step()

        # Update probe and baseline (minimize loss)
        # Note: loss is already computed, so we just do backward again
        optimizer_probe.zero_grad()
        optimizer_baseline.zero_grad()
        loss.backward()
        optimizer_probe.step()
        optimizer_baseline.step()

        return metrics

    def get_attributions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        history_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get attribution scores for input tokens (after training).

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            history_emb: Pre-computed history embedding [batch_size, embed_dim] (optional)

        Returns:
            attributions: Attribution scores [batch_size, seq_len]
        """
        self.eval()

        with torch.no_grad():
            news_encoder = self.model.news_encoder

            # Get embedding layer
            embedding_layer = None
            if hasattr(news_encoder, "bert"):
                embedding_layer = news_encoder.bert.embeddings
            elif hasattr(news_encoder, "lm"):
                embedding_layer = news_encoder.lm.embeddings
            elif hasattr(news_encoder, "embeddings"):
                embedding_layer = news_encoder.embeddings

            if embedding_layer is None:
                raise ValueError("Cannot find embedding layer")

            # Get embeddings and hidden states
            input_embeddings = embedding_layer(input_ids)

            if self.probe_type == "layerwise":
                hidden_states_stack = self.get_hidden_states_with_embeddings(
                    input_embeddings, attention_mask, news_encoder
                )
                hidden_states_for_probe = hidden_states_stack[:len(self.probe.probes)]
            else:
                hidden_states_for_probe = input_embeddings

            # Predict log-odds using probe
            if self.probe_type == "layerwise":
                log_alpha = self.probe(hidden_states_for_probe)
            else:
                log_alpha = self.probe(hidden_states_for_probe)

            # Get gates (deterministic mode)
            gates, _ = self.gates(log_alpha, training=False)

            # Mask out padding tokens
            gates = gates * attention_mask.float()

        return gates
