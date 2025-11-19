"""
TRAK (Tracing with the Randomly-projected After Kernel) Scorer.

Implements TRAK scoring to identify influential training samples.
Uses the MadryLab TRAK library: https://github.com/MadryLab/trak
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

try:
    from trak import TRAKer
    TRAK_AVAILABLE = True
except ImportError:
    TRAK_AVAILABLE = False
    print("WARNING: TRAK library not installed. Install with: pip install traker")


class TRAKScorer:
    """
    TRAK scorer for news recommendation models.

    Computes influence scores for training samples on model predictions.
    """

    def __init__(
        self,
        model,
        device='cuda',
        proj_dim=512,
        num_models=1,
        save_dir='./trak_results',
    ):
        """
        Initialize TRAK scorer.

        Args:
            model: PyTorch model to analyze
            device: Device to run on ('cuda' or 'cpu')
            proj_dim: Projection dimension for TRAK (default: 512)
            num_models: Number of models for ensembling (default: 1)
            save_dir: Directory to save TRAK results
        """
        if not TRAK_AVAILABLE:
            raise ImportError(
                "TRAK library is not installed. "
                "Install with: pip install traker"
            )

        self.model = model.to(device)
        self.device = device
        self.proj_dim = proj_dim
        self.num_models = num_models
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TRAKer
        self.traker = None

    def _get_model_output(self, batch, model):
        """
        Get model output (logits or scores) for a batch.

        Args:
            batch: Input batch
            model: Model to use

        Returns:
            Model output tensor
        """
        model.eval()
        with torch.no_grad():
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

            # Get model output
            output = model(batch)

            # Handle different output types
            if isinstance(output, dict):
                if 'logits' in output:
                    return output['logits']
                elif 'scores' in output:
                    return output['scores']

            return output

    def compute_gradients(self, dataloader, model_type='classification'):
        """
        Compute gradients for training samples.

        Args:
            dataloader: DataLoader for training data
            model_type: Type of model ('classification' or 'ranking')

        Returns:
            Gradients for all samples
        """
        print("\nComputing gradients for training samples...")

        gradients = []
        labels = []

        self.model.train()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing gradients")):
            # Get model output
            output = self._get_model_output(batch, self.model)

            # Get labels from impression data
            impression_data = batch.get('impression_data', None)
            if impression_data is not None:
                # Extract labels (real=0, fake=1)
                batch_labels = []
                for impressions in impression_data:
                    if len(impressions) > 0:
                        _, _, _, is_fake = impressions[0]
                        batch_labels.append(is_fake)
                labels.extend(batch_labels)

            # Compute loss
            if model_type == 'classification':
                # Binary classification loss
                if impression_data is not None:
                    target = torch.tensor(batch_labels, dtype=torch.float32).to(self.device)
                    loss = nn.BCEWithLogitsLoss()(output.squeeze(), target)
                else:
                    # Use dummy loss if no labels
                    loss = output.mean()
            else:
                # Ranking loss (can be customized)
                loss = -output.mean()

            # Compute gradients
            self.model.zero_grad()
            loss.backward()

            # Collect gradients
            batch_grads = []
            for param in self.model.parameters():
                if param.grad is not None:
                    batch_grads.append(param.grad.view(-1).cpu().detach().numpy())

            if batch_grads:
                gradients.append(np.concatenate(batch_grads))

        gradients = np.array(gradients)
        labels = np.array(labels)

        print(f"Computed gradients for {len(gradients)} samples")
        print(f"Gradient dimension: {gradients.shape[1] if len(gradients) > 0 else 0}")

        return gradients, labels

    def score_samples(self, train_loader, test_loader=None):
        """
        Compute TRAK scores for training samples.

        Args:
            train_loader: DataLoader for training data
            test_loader: Optional DataLoader for test data (for influence on test set)

        Returns:
            Dictionary with scores and metadata
        """
        print("\n" + "="*80)
        print("Computing TRAK Scores")
        print("="*80)

        # Compute training gradients
        train_grads, train_labels = self.compute_gradients(train_loader)

        # Project gradients to lower dimension
        print(f"\nProjecting gradients to dimension {self.proj_dim}...")
        from sklearn.random_projection import GaussianRandomProjection

        if train_grads.shape[1] > self.proj_dim:
            projector = GaussianRandomProjection(n_components=self.proj_dim)
            train_grads_proj = projector.fit_transform(train_grads)
        else:
            train_grads_proj = train_grads

        print(f"Projected gradient dimension: {train_grads_proj.shape[1]}")

        # Compute influence scores using simplified TRAK approximation
        # This is a simplified version - the full TRAK library provides more options
        print("\nComputing influence scores...")

        # Compute gradient covariance matrix
        Phi = train_grads_proj
        PhiT_Phi = Phi.T @ Phi

        # Add regularization for numerical stability
        reg = 1e-6
        PhiT_Phi_inv = np.linalg.inv(PhiT_Phi + reg * np.eye(PhiT_Phi.shape[0]))

        # Compute influence matrix: Phi @ PhiT_Phi_inv @ PhiT
        influence_matrix = Phi @ PhiT_Phi_inv @ Phi.T

        # Compute self-influence scores (diagonal of influence matrix)
        self_influence = np.diag(influence_matrix)

        # Normalize scores
        scores = (self_influence - self_influence.mean()) / (self_influence.std() + 1e-8)

        results = {
            'scores': scores,
            'labels': train_labels,
            'gradients': train_grads_proj,
            'influence_matrix': influence_matrix,
        }

        print(f"\nScore statistics:")
        print(f"  Mean: {scores.mean():.4f}")
        print(f"  Std: {scores.std():.4f}")
        print(f"  Min: {scores.min():.4f}")
        print(f"  Max: {scores.max():.4f}")

        return results


def compute_trak_scores(
    model,
    train_loader,
    device='cuda',
    proj_dim=512,
    save_dir='./trak_results',
):
    """
    Compute TRAK scores for training samples.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        device: Device to use
        proj_dim: Projection dimension
        save_dir: Directory to save results

    Returns:
        Dictionary with scores and metadata
    """
    scorer = TRAKScorer(
        model=model,
        device=device,
        proj_dim=proj_dim,
        save_dir=save_dir,
    )

    results = scorer.score_samples(train_loader)

    # Save results
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    np.save(save_path / 'trak_scores.npy', results['scores'])
    np.save(save_path / 'trak_labels.npy', results['labels'])
    np.save(save_path / 'trak_gradients.npy', results['gradients'])

    print(f"\nResults saved to: {save_path}")

    return results


def rank_training_samples(scores, labels, top_k=100):
    """
    Rank training samples by TRAK scores.

    Args:
        scores: TRAK scores for training samples
        labels: Labels for training samples (0=real, 1=fake)
        top_k: Number of top samples to return

    Returns:
        Dictionary with ranked samples
    """
    # Get indices sorted by score (descending)
    sorted_indices = np.argsort(scores)[::-1]

    # Get top-k most influential samples
    top_indices = sorted_indices[:top_k]
    top_scores = scores[top_indices]
    top_labels = labels[top_indices]

    # Get bottom-k least influential samples
    bottom_indices = sorted_indices[-top_k:]
    bottom_scores = scores[bottom_indices]
    bottom_labels = labels[bottom_indices]

    # Separate by label
    fake_mask = labels == 1
    real_mask = labels == 0

    fake_scores = scores[fake_mask]
    real_scores = scores[real_mask]

    fake_indices = np.where(fake_mask)[0]
    real_indices = np.where(real_mask)[0]

    # Sort within each class
    fake_sorted = fake_indices[np.argsort(fake_scores)[::-1]]
    real_sorted = real_indices[np.argsort(real_scores)[::-1]]

    results = {
        'top_k_indices': top_indices,
        'top_k_scores': top_scores,
        'top_k_labels': top_labels,
        'bottom_k_indices': bottom_indices,
        'bottom_k_scores': bottom_scores,
        'bottom_k_labels': bottom_labels,
        'top_fake_indices': fake_sorted[:top_k],
        'top_fake_scores': fake_scores[np.argsort(fake_scores)[::-1][:top_k]],
        'top_real_indices': real_sorted[:top_k],
        'top_real_scores': real_scores[np.argsort(real_scores)[::-1][:top_k]],
    }

    print("\n" + "="*80)
    print("Ranking Summary")
    print("="*80)
    print(f"\nTop-{top_k} most influential samples:")
    print(f"  Fake news: {np.sum(top_labels == 1)} ({np.sum(top_labels == 1)/len(top_labels)*100:.1f}%)")
    print(f"  Real news: {np.sum(top_labels == 0)} ({np.sum(top_labels == 0)/len(top_labels)*100:.1f}%)")

    print(f"\nBottom-{top_k} least influential samples:")
    print(f"  Fake news: {np.sum(bottom_labels == 1)} ({np.sum(bottom_labels == 1)/len(bottom_labels)*100:.1f}%)")
    print(f"  Real news: {np.sum(bottom_labels == 0)} ({np.sum(bottom_labels == 0)/len(bottom_labels)*100:.1f}%)")

    return results
