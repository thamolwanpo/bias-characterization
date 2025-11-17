"""
Stochastic Gates for DIFFMASK.

Implements Hard Concrete distribution for sparse, differentiable masking.
Reference: https://arxiv.org/abs/1712.01312 (Louizos et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class HardConcrete(nn.Module):
    """
    Hard Concrete distribution for stochastic gating.

    This distribution allows for sparse, differentiable sampling where gates
    can be exactly 0 or 1, making it suitable for feature selection.

    The distribution is a stretched and rectified version of Binary Concrete:
    - Binary Concrete: continuous relaxation of Bernoulli
    - Hard Concrete: stretched to [0, 1] and clipped, allowing exact 0/1 values
    """

    def __init__(
        self,
        temperature: float = 2.0 / 3.0,
        stretch: float = 0.1,
        eps: float = 1e-8
    ):
        """
        Args:
            temperature: Temperature parameter for relaxation (lower = more discrete)
            stretch: Stretch parameter (determines range before clipping)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.temperature = temperature
        self.stretch = stretch
        self.eps = eps

        # Compute stretch bounds
        self.gamma = -stretch
        self.zeta = 1.0 + stretch

    def sample_hard_concrete(
        self,
        log_alpha: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """
        Sample from Hard Concrete distribution.

        Args:
            log_alpha: Log-odds parameters [batch_size, n_tokens] or [..., n_tokens]
            training: If True, sample stochastically; if False, use deterministic mode

        Returns:
            gates: Sampled gates in [0, 1] [..., n_tokens]
        """
        if training:
            # Sample from uniform distribution
            u = torch.rand_like(log_alpha)
            u = torch.clamp(u, self.eps, 1.0 - self.eps)

            # Reparametrization trick: s = sigmoid((log(u) - log(1-u) + log_alpha) / temp)
            s = torch.sigmoid(
                (torch.log(u) - torch.log(1.0 - u) + log_alpha) / self.temperature
            )
        else:
            # Deterministic mode: use expected value
            s = torch.sigmoid(log_alpha)

        # Stretch and rectify to get Hard Concrete
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        z = torch.clamp(s_bar, 0.0, 1.0)

        return z

    def expected_l0(self, log_alpha: torch.Tensor) -> torch.Tensor:
        """
        Compute expected L0 norm (number of non-zero gates).

        Args:
            log_alpha: Log-odds parameters [..., n_tokens]

        Returns:
            expected_l0: Expected number of non-zero elements [...]
        """
        # Probability that gate is non-zero
        # P(z > 0) = P(s_bar > 0) = P(s > -gamma / (zeta - gamma))
        threshold = -self.gamma / (self.zeta - self.gamma)

        # Convert threshold to tensor with same dtype and device as log_alpha
        threshold_tensor = torch.tensor(threshold, dtype=log_alpha.dtype, device=log_alpha.device)

        # P(s > threshold) = sigmoid((log_alpha - logit(threshold)) / temperature)
        logit_threshold = torch.log(threshold_tensor / (1.0 - threshold_tensor + self.eps))
        p_nonzero = torch.sigmoid((log_alpha - logit_threshold) / self.temperature)

        # Sum over token dimension to get expected L0
        return p_nonzero.sum(dim=-1)

    def forward(
        self,
        log_alpha: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: sample gates and compute expected L0.

        Args:
            log_alpha: Log-odds parameters [..., n_tokens]
            training: Training mode flag

        Returns:
            gates: Sampled gates [..., n_tokens]
            expected_l0: Expected L0 norm [...]
        """
        gates = self.sample_hard_concrete(log_alpha, training=training)
        l0 = self.expected_l0(log_alpha)

        return gates, l0


class StochasticGates(nn.Module):
    """
    Learnable stochastic gates for token selection.

    This module learns log-odds parameters (log_alpha) for each token position
    and samples binary gates using the Hard Concrete distribution.
    """

    def __init__(
        self,
        input_dim: int,
        temperature: float = 2.0 / 3.0,
        stretch: float = 0.1
    ):
        """
        Args:
            input_dim: Dimension of input features (used for gate prediction)
            temperature: Temperature for Hard Concrete
            stretch: Stretch parameter for Hard Concrete
        """
        super().__init__()

        # Hard Concrete distribution
        self.hard_concrete = HardConcrete(
            temperature=temperature,
            stretch=stretch
        )

        # Initialize log_alpha parameters (will be predicted by probe network)
        # These are not learned directly but predicted from hidden states

    def forward(
        self,
        log_alpha: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample gates from log-odds parameters.

        Args:
            log_alpha: Log-odds parameters from probe network [..., n_tokens]
            training: Training mode flag

        Returns:
            gates: Sampled gates [..., n_tokens]
            expected_l0: Expected L0 norm [...]
        """
        return self.hard_concrete(log_alpha, training=training)
