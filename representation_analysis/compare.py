"""
Compare representations between clean and poisoned models.
Embedding space analysis with scores and visualizations.

UPDATED:
- Fixed import to use updated representation extraction
- Added --n_samples parameter to limit analysis
"""

import argparse
import yaml
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import ks_2samp
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

import sys
import os

# sys.path.insert(
#     0,
#     os.path.abspath(
#         "/content/drive/MyDrive/bias-characterized/bias-characterization/plm4newsrs"
#     ),
# )

sys.path.insert(
    0,
    os.path.abspath("/home/thamo/PhD/bias-characterization/plm4newsrs"),
)

from configs import load_config as load_model_config
from data_loader import load_test_data, get_data_statistics_fast
from representation import extract_news_representations  # UPDATED IMPORT


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_limited_dataloader(dataset, data_loader, n_samples=None):
    """
    Create a limited dataloader if n_samples is specified.

    Args:
        dataset: Original dataset
        data_loader: Original dataloader
        n_samples: Number of samples to use (None = use all)

    Returns:
        Limited dataloader or original if n_samples is None
    """
    if n_samples is None:
        return data_loader

    from torch.utils.data import Subset, DataLoader
    from data_loader import benchmark_collate_fn

    # Create a subset of the dataset
    indices = list(range(min(n_samples, len(dataset))))
    subset = Subset(dataset, indices)

    # Create new dataloader with subset
    limited_loader = DataLoader(
        subset,
        batch_size=data_loader.batch_size,
        shuffle=False,
        collate_fn=benchmark_collate_fn,
        num_workers=0,  # Keep at 0 for stability
    )

    print(
        f"Created limited dataloader with {len(subset)} samples (out of {len(dataset)} total)"
    )
    return limited_loader


def extract_deduplicated_embeddings(embeddings, labels, ids):
    """
    Extract and deduplicate embeddings.

    Args:
        embeddings: torch.Tensor of shape [n_samples, n_candidates, embedding_dim] or [n_samples, embedding_dim]
        labels: numpy array of shape [n_samples, n_candidates] or [n_samples]
        ids: numpy array of IDs for deduplication

    Returns:
        embeddings_np: numpy array of shape [n_unique, embedding_dim]
        labels_np: numpy array of shape [n_unique] with 'real'/'fake' strings
    """
    import torch

    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    # Handle 3D embeddings (news embeddings with candidates)
    if len(embeddings.shape) == 3:
        # Take only first candidate
        embeddings = embeddings[:, 0, :]
        labels = labels[:, 0]
        if ids is not None:
            ids = ids[:, 0]

    # Deduplicate by IDs if provided
    if ids is not None:
        unique_ids, unique_indices = np.unique(ids, return_index=True)
        embeddings = embeddings[unique_indices]
        labels = labels[unique_indices]

    # Convert labels to string format
    labels_str = np.array(["fake" if label == 1 else "real" for label in labels])

    return embeddings, labels_str


def compute_embedding_stats(embeddings, labels, model_name):
    """
    Compute various statistics about embeddings.

    Args:
        embeddings: numpy array of shape [n_samples, embedding_dim]
        labels: numpy array of shape [n_samples] with 'real'/'fake' strings
        model_name: string identifier for the model

    Returns:
        dict with statistics
    """
    stats = {}

    # 1. Embedding Norms
    norms = np.linalg.norm(embeddings, axis=1)
    stats["mean_norm"] = float(np.mean(norms))
    stats["std_norm"] = float(np.std(norms))

    real_mask = labels == "real"
    fake_mask = labels == "fake"

    stats["mean_norm_real"] = float(np.mean(norms[real_mask]))
    stats["mean_norm_fake"] = float(np.mean(norms[fake_mask]))

    # 2. Centroid Distance
    real_centroid = np.mean(embeddings[real_mask], axis=0)
    fake_centroid = np.mean(embeddings[fake_mask], axis=0)
    stats["centroid_distance"] = float(np.linalg.norm(real_centroid - fake_centroid))

    # 3. Silhouette Score (separation quality)
    label_numeric = (labels == "fake").astype(int)
    stats["silhouette_score"] = float(silhouette_score(embeddings, label_numeric))

    # 4. Variance per class
    stats["variance_real"] = float(np.mean(np.var(embeddings[real_mask], axis=0)))
    stats["variance_fake"] = float(np.mean(np.var(embeddings[fake_mask], axis=0)))

    # 5. Statistical test for norm distributions
    ks_statistic, ks_pvalue = ks_2samp(norms[real_mask], norms[fake_mask])
    stats["norm_ks_statistic"] = float(ks_statistic)
    stats["norm_ks_pvalue"] = float(ks_pvalue)

    return stats


def plot_embedding_norms(
    clean_emb, poisoned_emb, clean_labels, poisoned_labels, save_path, title
):
    """
    Plot norm distributions for clean vs poisoned models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Clean model
    clean_norms_real = np.linalg.norm(clean_emb[clean_labels == "real"], axis=1)
    clean_norms_fake = np.linalg.norm(clean_emb[clean_labels == "fake"], axis=1)

    axes[0].hist(clean_norms_real, bins=50, alpha=0.6, label="Real", color="blue")
    axes[0].hist(clean_norms_fake, bins=50, alpha=0.6, label="Fake", color="red")
    axes[0].set_xlabel("L2 Norm", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].set_title(f"Clean Model - {title}", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].axvline(
        np.mean(clean_norms_real),
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Mean Real",
    )
    axes[0].axvline(
        np.mean(clean_norms_fake),
        color="red",
        linestyle="--",
        linewidth=2,
        label="Mean Fake",
    )
    axes[0].grid(alpha=0.3)

    # Poisoned model
    poisoned_norms_real = np.linalg.norm(
        poisoned_emb[poisoned_labels == "real"], axis=1
    )
    poisoned_norms_fake = np.linalg.norm(
        poisoned_emb[poisoned_labels == "fake"], axis=1
    )

    axes[1].hist(poisoned_norms_real, bins=50, alpha=0.6, label="Real", color="blue")
    axes[1].hist(poisoned_norms_fake, bins=50, alpha=0.6, label="Fake", color="red")
    axes[1].set_xlabel("L2 Norm", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].set_title(f"Poisoned Model - {title}", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].axvline(
        np.mean(poisoned_norms_real), color="blue", linestyle="--", linewidth=2
    )
    axes[1].axvline(
        np.mean(poisoned_norms_fake), color="red", linestyle="--", linewidth=2
    )
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pca_comparison(
    clean_emb, poisoned_emb, clean_labels, poisoned_labels, save_path, title
):
    """
    Side-by-side PCA comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Clean model PCA
    pca_clean = PCA(n_components=2)
    clean_reduced = pca_clean.fit_transform(clean_emb)

    # Compute centroids in PCA space
    real_mask_clean = clean_labels == "real"
    fake_mask_clean = clean_labels == "fake"
    clean_real_centroid = np.mean(clean_reduced[real_mask_clean], axis=0)
    clean_fake_centroid = np.mean(clean_reduced[fake_mask_clean], axis=0)

    for label, color in [("real", "blue"), ("fake", "red")]:
        mask = clean_labels == label
        axes[0].scatter(
            clean_reduced[mask, 0],
            clean_reduced[mask, 1],
            c=color,
            alpha=0.6,
            label=label.capitalize(),
            s=20,
        )

    # Plot centroids
    axes[0].scatter(
        clean_real_centroid[0],
        clean_real_centroid[1],
        c="blue",
        marker="*",
        s=500,
        edgecolors="black",
        linewidths=2,
        label="Real Centroid",
        zorder=5,
    )
    axes[0].scatter(
        clean_fake_centroid[0],
        clean_fake_centroid[1],
        c="red",
        marker="*",
        s=500,
        edgecolors="black",
        linewidths=2,
        label="Fake Centroid",
        zorder=5,
    )

    # Draw line between centroids
    axes[0].plot(
        [clean_real_centroid[0], clean_fake_centroid[0]],
        [clean_real_centroid[1], clean_fake_centroid[1]],
        "k--",
        linewidth=2,
        alpha=0.5,
        label="Centroid Distance",
    )

    axes[0].set_xlabel(
        f"PC1 ({pca_clean.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11
    )
    axes[0].set_ylabel(
        f"PC2 ({pca_clean.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11
    )
    axes[0].set_title(f"Clean Model - {title}", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Poisoned model PCA
    pca_poisoned = PCA(n_components=2)
    poisoned_reduced = pca_poisoned.fit_transform(poisoned_emb)

    # Compute centroids in PCA space
    real_mask_poisoned = poisoned_labels == "real"
    fake_mask_poisoned = poisoned_labels == "fake"
    poisoned_real_centroid = np.mean(poisoned_reduced[real_mask_poisoned], axis=0)
    poisoned_fake_centroid = np.mean(poisoned_reduced[fake_mask_poisoned], axis=0)

    for label, color in [("real", "blue"), ("fake", "red")]:
        mask = poisoned_labels == label
        axes[1].scatter(
            poisoned_reduced[mask, 0],
            poisoned_reduced[mask, 1],
            c=color,
            alpha=0.6,
            label=label.capitalize(),
            s=20,
        )

    # Plot centroids
    axes[1].scatter(
        poisoned_real_centroid[0],
        poisoned_real_centroid[1],
        c="blue",
        marker="*",
        s=500,
        edgecolors="black",
        linewidths=2,
        label="Real Centroid",
        zorder=5,
    )
    axes[1].scatter(
        poisoned_fake_centroid[0],
        poisoned_fake_centroid[1],
        c="red",
        marker="*",
        s=500,
        edgecolors="black",
        linewidths=2,
        label="Fake Centroid",
        zorder=5,
    )

    # Draw line between centroids
    axes[1].plot(
        [poisoned_real_centroid[0], poisoned_fake_centroid[0]],
        [poisoned_real_centroid[1], poisoned_fake_centroid[1]],
        "k--",
        linewidth=2,
        alpha=0.5,
        label="Centroid Distance",
    )

    axes[1].set_xlabel(
        f"PC1 ({pca_poisoned.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11
    )
    axes[1].set_ylabel(
        f"PC2 ({pca_poisoned.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11
    )
    axes[1].set_title(f"Poisoned Model - {title}", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_tsne_comparison(
    clean_emb,
    poisoned_emb,
    clean_labels,
    poisoned_labels,
    save_path,
    title,
    perplexity=30,
    random_state=42,
):
    """
    Side-by-side t-SNE comparison with centroids.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Clean model t-SNE
    print(f"    Computing t-SNE for clean model (perplexity={perplexity})...")
    tsne_clean = TSNE(
        n_components=2, perplexity=perplexity, random_state=random_state, max_iter=1000
    )
    clean_reduced = tsne_clean.fit_transform(clean_emb)

    # Compute centroids in t-SNE space
    real_mask_clean = clean_labels == "real"
    fake_mask_clean = clean_labels == "fake"
    clean_real_centroid = np.mean(clean_reduced[real_mask_clean], axis=0)
    clean_fake_centroid = np.mean(clean_reduced[fake_mask_clean], axis=0)

    for label, color in [("real", "blue"), ("fake", "red")]:
        mask = clean_labels == label
        axes[0].scatter(
            clean_reduced[mask, 0],
            clean_reduced[mask, 1],
            c=color,
            alpha=0.6,
            label=label.capitalize(),
            s=20,
        )

    # Plot centroids
    axes[0].scatter(
        clean_real_centroid[0],
        clean_real_centroid[1],
        c="blue",
        marker="*",
        s=500,
        edgecolors="black",
        linewidths=2,
        label="Real Centroid",
        zorder=5,
    )
    axes[0].scatter(
        clean_fake_centroid[0],
        clean_fake_centroid[1],
        c="red",
        marker="*",
        s=500,
        edgecolors="black",
        linewidths=2,
        label="Fake Centroid",
        zorder=5,
    )

    # Draw line between centroids
    axes[0].plot(
        [clean_real_centroid[0], clean_fake_centroid[0]],
        [clean_real_centroid[1], clean_fake_centroid[1]],
        "k--",
        linewidth=2,
        alpha=0.5,
        label="Centroid Distance",
    )

    axes[0].set_xlabel("t-SNE Dimension 1", fontsize=11)
    axes[0].set_ylabel("t-SNE Dimension 2", fontsize=11)
    axes[0].set_title(f"Clean Model - {title}", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Poisoned model t-SNE
    print(f"    Computing t-SNE for poisoned model (perplexity={perplexity})...")
    tsne_poisoned = TSNE(
        n_components=2, perplexity=perplexity, random_state=random_state, max_iter=1000
    )
    poisoned_reduced = tsne_poisoned.fit_transform(poisoned_emb)

    # Compute centroids in t-SNE space
    real_mask_poisoned = poisoned_labels == "real"
    fake_mask_poisoned = poisoned_labels == "fake"
    poisoned_real_centroid = np.mean(poisoned_reduced[real_mask_poisoned], axis=0)
    poisoned_fake_centroid = np.mean(poisoned_reduced[fake_mask_poisoned], axis=0)

    for label, color in [("real", "blue"), ("fake", "red")]:
        mask = poisoned_labels == label
        axes[1].scatter(
            poisoned_reduced[mask, 0],
            poisoned_reduced[mask, 1],
            c=color,
            alpha=0.6,
            label=label.capitalize(),
            s=20,
        )

    # Plot centroids
    axes[1].scatter(
        poisoned_real_centroid[0],
        poisoned_real_centroid[1],
        c="blue",
        marker="*",
        s=500,
        edgecolors="black",
        linewidths=2,
        label="Real Centroid",
        zorder=5,
    )
    axes[1].scatter(
        poisoned_fake_centroid[0],
        poisoned_fake_centroid[1],
        c="red",
        marker="*",
        s=500,
        edgecolors="black",
        linewidths=2,
        label="Fake Centroid",
        zorder=5,
    )

    # Draw line between centroids
    axes[1].plot(
        [poisoned_real_centroid[0], poisoned_fake_centroid[0]],
        [poisoned_real_centroid[1], poisoned_fake_centroid[1]],
        "k--",
        linewidth=2,
        alpha=0.5,
        label="Centroid Distance",
    )

    axes[1].set_xlabel("t-SNE Dimension 1", fontsize=11)
    axes[1].set_ylabel("t-SNE Dimension 2", fontsize=11)
    axes[1].set_title(f"Poisoned Model - {title}", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_dimension_stats(
    clean_emb, poisoned_emb, clean_labels, poisoned_labels, save_path, title
):
    """
    Plot mean and std per dimension.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    n_dims = min(50, clean_emb.shape[1])
    dims = range(n_dims)

    # Clean model - mean per dimension
    clean_mean_real = np.mean(clean_emb[clean_labels == "real"], axis=0)[:n_dims]
    clean_mean_fake = np.mean(clean_emb[clean_labels == "fake"], axis=0)[:n_dims]

    axes[0, 0].plot(dims, clean_mean_real, label="Real", color="blue", linewidth=2)
    axes[0, 0].plot(dims, clean_mean_fake, label="Fake", color="red", linewidth=2)
    axes[0, 0].set_xlabel("Dimension", fontsize=11)
    axes[0, 0].set_ylabel("Mean Value", fontsize=11)
    axes[0, 0].set_title(
        f"Clean Model - Mean per Dimension", fontsize=12, fontweight="bold"
    )
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Poisoned model - mean per dimension
    poisoned_mean_real = np.mean(poisoned_emb[poisoned_labels == "real"], axis=0)[
        :n_dims
    ]
    poisoned_mean_fake = np.mean(poisoned_emb[poisoned_labels == "fake"], axis=0)[
        :n_dims
    ]

    axes[0, 1].plot(dims, poisoned_mean_real, label="Real", color="blue", linewidth=2)
    axes[0, 1].plot(dims, poisoned_mean_fake, label="Fake", color="red", linewidth=2)
    axes[0, 1].set_xlabel("Dimension", fontsize=11)
    axes[0, 1].set_ylabel("Mean Value", fontsize=11)
    axes[0, 1].set_title(
        f"Poisoned Model - Mean per Dimension", fontsize=12, fontweight="bold"
    )
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Clean model - std per dimension
    clean_std_real = np.std(clean_emb[clean_labels == "real"], axis=0)[:n_dims]
    clean_std_fake = np.std(clean_emb[clean_labels == "fake"], axis=0)[:n_dims]

    axes[1, 0].plot(dims, clean_std_real, label="Real", color="blue", linewidth=2)
    axes[1, 0].plot(dims, clean_std_fake, label="Fake", color="red", linewidth=2)
    axes[1, 0].set_xlabel("Dimension", fontsize=11)
    axes[1, 0].set_ylabel("Std Deviation", fontsize=11)
    axes[1, 0].set_title(
        f"Clean Model - Std per Dimension", fontsize=12, fontweight="bold"
    )
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Poisoned model - std per dimension
    poisoned_std_real = np.std(poisoned_emb[poisoned_labels == "real"], axis=0)[:n_dims]
    poisoned_std_fake = np.std(poisoned_emb[poisoned_labels == "fake"], axis=0)[:n_dims]

    axes[1, 1].plot(dims, poisoned_std_real, label="Real", color="blue", linewidth=2)
    axes[1, 1].plot(dims, poisoned_std_fake, label="Fake", color="red", linewidth=2)
    axes[1, 1].set_xlabel("Dimension", fontsize=11)
    axes[1, 1].set_ylabel("Std Deviation", fontsize=11)
    axes[1, 1].set_title(
        f"Poisoned Model - Std per Dimension", fontsize=12, fontweight="bold"
    )
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_separation_metrics(clean_stats, poisoned_stats, embedding_type, save_path):
    """
    Bar plot comparing separation metrics.
    """
    metrics = [
        "centroid_distance",
        "silhouette_score",
        "mean_norm_real",
        "mean_norm_fake",
    ]
    metric_labels = [
        "Centroid\nDistance",
        "Silhouette\nScore",
        "Mean Norm\n(Real)",
        "Mean Norm\n(Fake)",
    ]
    clean_values = [clean_stats[m] for m in metrics]
    poisoned_values = [poisoned_stats[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width / 2,
        clean_values,
        width,
        label="Clean Model",
        color="green",
        alpha=0.7,
    )
    ax.bar(
        x + width / 2,
        poisoned_values,
        width,
        label="Poisoned Model",
        color="orange",
        alpha=0.7,
    )

    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(
        f"{embedding_type} Embedding - Separation Metrics Comparison",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def print_comparison_report(clean_stats, poisoned_stats, embedding_type):
    """
    Print detailed comparison report.
    """
    print(f"\n{'='*75}")
    print(f"{embedding_type.upper()} EMBEDDING STATISTICS")
    print(f"{'='*75}\n")

    print(f"{'Metric':<30} {'Clean':<15} {'Poisoned':<15} {'Change':<15}")
    print("-" * 75)

    for key in clean_stats.keys():
        clean_val = clean_stats[key]
        poisoned_val = poisoned_stats[key]

        if clean_val != 0:
            change_pct = ((poisoned_val - clean_val) / abs(clean_val)) * 100
            change_str = f"{change_pct:+.2f}%"
        else:
            change_str = "N/A"

        print(f"{key:<30} {clean_val:<15.4f} {poisoned_val:<15.4f} {change_str:<15}")

    print("\n" + "=" * 75)

    # Key findings
    print("\nKEY FINDINGS:")

    # Check centroid distance
    centroid_change = (
        poisoned_stats["centroid_distance"] - clean_stats["centroid_distance"]
    ) / clean_stats["centroid_distance"]
    if centroid_change > 0.1:
        print(
            f"  ✓ Centroid distance INCREASED by {centroid_change*100:.1f}% - classes more separated"
        )
    elif centroid_change < -0.1:
        print(
            f"  ⚠ Centroid distance DECREASED by {abs(centroid_change)*100:.1f}% - classes closer together"
        )

    # Check silhouette score
    silhouette_change = (
        poisoned_stats["silhouette_score"] - clean_stats["silhouette_score"]
    )
    if abs(silhouette_change) > 0.05:
        direction = "INCREASED" if silhouette_change > 0 else "DECREASED"
        symbol = "✓" if silhouette_change > 0 else "⚠"
        quality = "better" if silhouette_change > 0 else "worse"
        print(
            f"  {symbol} Silhouette score {direction} by {abs(silhouette_change):.3f} - {quality} class separation"
        )

    # Check norms
    clean_norm_gap = abs(clean_stats["mean_norm_fake"] - clean_stats["mean_norm_real"])
    poisoned_norm_gap = abs(
        poisoned_stats["mean_norm_fake"] - poisoned_stats["mean_norm_real"]
    )
    if poisoned_norm_gap > clean_norm_gap * 1.2:
        print(
            f"  ⚠ Norm gap between real/fake INCREASED from {clean_norm_gap:.3f} to {poisoned_norm_gap:.3f}"
        )

    # Check statistical significance
    if poisoned_stats["norm_ks_pvalue"] < 0.05:
        print(
            f"  ✓ Norm distributions are SIGNIFICANTLY different (p = {poisoned_stats['norm_ks_pvalue']:.4f})"
        )

    print()


def analyze_user_preferences(user_embeddings, news_embeddings, news_labels, model_name):
    """
    Determine if users prefer real or fake news regions.

    Method 1: User-Centroid Alignment Analysis

    Args:
        user_embeddings: numpy array of shape [n_users, embedding_dim]
        news_embeddings: numpy array of shape [n_news, embedding_dim]
        news_labels: numpy array of shape [n_news] with 'real'/'fake' strings
        model_name: string identifier for the model

    Returns:
        dict with analysis results
    """
    # Separate news embeddings by label
    real_mask = news_labels == "real"
    fake_mask = news_labels == "fake"

    news_embeddings_real = news_embeddings[real_mask]
    news_embeddings_fake = news_embeddings[fake_mask]

    # Compute centroids
    real_centroid = np.mean(news_embeddings_real, axis=0)
    fake_centroid = np.mean(news_embeddings_fake, axis=0)
    mean_user = np.mean(user_embeddings, axis=0)

    # Compute alignment scores (dot product - what the model actually uses)
    alignment_real = []
    alignment_fake = []

    for user_emb in user_embeddings:
        score_real = np.dot(user_emb, real_centroid)
        score_fake = np.dot(user_emb, fake_centroid)

        alignment_real.append(score_real)
        alignment_fake.append(score_fake)

    # Also compute cosine similarity (normalized version)
    cos_real = cosine_similarity(
        user_embeddings, real_centroid.reshape(1, -1)
    ).flatten()
    cos_fake = cosine_similarity(
        user_embeddings, fake_centroid.reshape(1, -1)
    ).flatten()

    # Compute statistics
    results = {
        "mean_alignment_real": float(np.mean(alignment_real)),
        "mean_alignment_fake": float(np.mean(alignment_fake)),
        "mean_cos_sim_real": float(np.mean(cos_real)),
        "mean_cos_sim_fake": float(np.mean(cos_fake)),
        "prefer_fake_count": int(
            np.sum(np.array(alignment_fake) > np.array(alignment_real))
        ),
        "prefer_real_count": int(
            np.sum(np.array(alignment_real) > np.array(alignment_fake))
        ),
        "alignment_real_all": alignment_real,
        "alignment_fake_all": alignment_fake,
        "real_centroid": real_centroid,
        "fake_centroid": fake_centroid,
        "mean_user": mean_user,
    }

    return results


def plot_user_centroid_alignment(clean_results, poisoned_results, save_path):
    """
    Visualize how users align with real vs fake centroids.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Distribution of alignment scores (Clean)
    axes[0, 0].hist(
        clean_results["alignment_real_all"],
        bins=50,
        alpha=0.6,
        label="Real Centroid",
        color="blue",
    )
    axes[0, 0].hist(
        clean_results["alignment_fake_all"],
        bins=50,
        alpha=0.6,
        label="Fake Centroid",
        color="red",
    )
    axes[0, 0].axvline(
        clean_results["mean_alignment_real"], color="blue", linestyle="--", linewidth=2
    )
    axes[0, 0].axvline(
        clean_results["mean_alignment_fake"], color="red", linestyle="--", linewidth=2
    )
    axes[0, 0].set_xlabel("Alignment Score (Dot Product)", fontsize=11)
    axes[0, 0].set_ylabel("Frequency", fontsize=11)
    axes[0, 0].set_title(
        "Clean Model - User Alignment Distributions", fontsize=12, fontweight="bold"
    )
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Distribution of alignment scores (Poisoned)
    axes[0, 1].hist(
        poisoned_results["alignment_real_all"],
        bins=50,
        alpha=0.6,
        label="Real Centroid",
        color="blue",
    )
    axes[0, 1].hist(
        poisoned_results["alignment_fake_all"],
        bins=50,
        alpha=0.6,
        label="Fake Centroid",
        color="red",
    )
    axes[0, 1].axvline(
        poisoned_results["mean_alignment_real"],
        color="blue",
        linestyle="--",
        linewidth=2,
    )
    axes[0, 1].axvline(
        poisoned_results["mean_alignment_fake"],
        color="red",
        linestyle="--",
        linewidth=2,
    )
    axes[0, 1].set_xlabel("Alignment Score (Dot Product)", fontsize=11)
    axes[0, 1].set_ylabel("Frequency", fontsize=11)
    axes[0, 1].set_title(
        "Poisoned Model - User Alignment Distributions", fontsize=12, fontweight="bold"
    )
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Scatter plot (Clean) - Each user's preference
    diff_clean = np.array(clean_results["alignment_real_all"]) - np.array(
        clean_results["alignment_fake_all"]
    )
    axes[1, 0].scatter(range(len(diff_clean)), diff_clean, alpha=0.5, s=20)
    axes[1, 0].axhline(0, color="black", linestyle="--", linewidth=2)
    axes[1, 0].fill_between(
        range(len(diff_clean)),
        0,
        diff_clean,
        where=(diff_clean > 0),
        alpha=0.3,
        color="blue",
        label="Prefer Real",
    )
    axes[1, 0].fill_between(
        range(len(diff_clean)),
        0,
        diff_clean,
        where=(diff_clean < 0),
        alpha=0.3,
        color="red",
        label="Prefer Fake",
    )
    axes[1, 0].set_xlabel("User Index", fontsize=11)
    axes[1, 0].set_ylabel("Preference Gap (Real - Fake)", fontsize=11)
    axes[1, 0].set_title(
        "Clean Model - Per-User Preferences", fontsize=12, fontweight="bold"
    )
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Scatter plot (Poisoned) - Each user's preference
    diff_poisoned = np.array(poisoned_results["alignment_real_all"]) - np.array(
        poisoned_results["alignment_fake_all"]
    )
    axes[1, 1].scatter(range(len(diff_poisoned)), diff_poisoned, alpha=0.5, s=20)
    axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=2)
    axes[1, 1].fill_between(
        range(len(diff_poisoned)),
        0,
        diff_poisoned,
        where=(diff_poisoned > 0),
        alpha=0.3,
        color="blue",
        label="Prefer Real",
    )
    axes[1, 1].fill_between(
        range(len(diff_poisoned)),
        0,
        diff_poisoned,
        where=(diff_poisoned < 0),
        alpha=0.3,
        color="red",
        label="Prefer Fake",
    )
    axes[1, 1].set_xlabel("User Index", fontsize=11)
    axes[1, 1].set_ylabel("Preference Gap (Real - Fake)", fontsize=11)
    axes[1, 1].set_title(
        "Poisoned Model - Per-User Preferences", fontsize=12, fontweight="bold"
    )
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def compute_actual_scores(user_embeddings, news_embeddings, news_labels):
    """
    Compute actual user-news dot product scores (ground truth).

    Args:
        user_embeddings: numpy array of shape [n_users, embedding_dim]
        news_embeddings: numpy array of shape [n_news, embedding_dim]
        news_labels: numpy array of shape [n_news] with 'real'/'fake' strings

    Returns:
        scores_real: numpy array of scores for real news
        scores_fake: numpy array of scores for fake news
    """
    scores_real = []
    scores_fake = []

    real_mask = news_labels == "real"
    fake_mask = news_labels == "fake"

    news_embeddings_real = news_embeddings[real_mask]
    news_embeddings_fake = news_embeddings[fake_mask]

    # Compute all user-news scores
    for user_emb in user_embeddings:
        # Scores with real news
        for news_emb in news_embeddings_real:
            score = np.dot(user_emb, news_emb)
            scores_real.append(score)

        # Scores with fake news
        for news_emb in news_embeddings_fake:
            score = np.dot(user_emb, news_emb)
            scores_fake.append(score)

    return np.array(scores_real), np.array(scores_fake)


def plot_actual_recommendation_scores(
    clean_scores_real,
    clean_scores_fake,
    poisoned_scores_real,
    poisoned_scores_fake,
    save_path,
):
    """
    Visualize actual recommendation scores distribution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Clean model
    axes[0].hist(clean_scores_real, bins=50, alpha=0.6, label="Real News", color="blue")
    axes[0].hist(clean_scores_fake, bins=50, alpha=0.6, label="Fake News", color="red")
    axes[0].axvline(
        np.mean(clean_scores_real),
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Mean Real: {np.mean(clean_scores_real):.3f}",
    )
    axes[0].axvline(
        np.mean(clean_scores_fake),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean Fake: {np.mean(clean_scores_fake):.3f}",
    )
    axes[0].set_xlabel("Recommendation Score", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].set_title(
        "Clean Model - Actual Recommendation Scores", fontsize=12, fontweight="bold"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Poisoned model
    axes[1].hist(
        poisoned_scores_real, bins=50, alpha=0.6, label="Real News", color="blue"
    )
    axes[1].hist(
        poisoned_scores_fake, bins=50, alpha=0.6, label="Fake News", color="red"
    )
    axes[1].axvline(
        np.mean(poisoned_scores_real),
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Mean Real: {np.mean(poisoned_scores_real):.3f}",
    )
    axes[1].axvline(
        np.mean(poisoned_scores_fake),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean Fake: {np.mean(poisoned_scores_fake):.3f}",
    )
    axes[1].set_xlabel("Recommendation Score", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].set_title(
        "Poisoned Model - Actual Recommendation Scores", fontsize=12, fontweight="bold"
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def print_user_preference_analysis(
    clean_results,
    poisoned_results,
    clean_scores_real,
    clean_scores_fake,
    poisoned_scores_real,
    poisoned_scores_fake,
):
    """
    Print comprehensive user preference analysis report.
    """
    print("\n" + "=" * 75)
    print("USER PREFERENCE ANALYSIS")
    print("=" * 75)

    print(f"\n{'Metric':<45} {'Clean':<15} {'Poisoned':<15}")
    print("-" * 75)

    print(
        f"{'Mean alignment with REAL centroid':<45} "
        f"{clean_results['mean_alignment_real']:<15.4f} "
        f"{poisoned_results['mean_alignment_real']:<15.4f}"
    )

    print(
        f"{'Mean alignment with FAKE centroid':<45} "
        f"{clean_results['mean_alignment_fake']:<15.4f} "
        f"{poisoned_results['mean_alignment_fake']:<15.4f}"
    )

    print(
        f"\n{'Mean cosine similarity with REAL':<45} "
        f"{clean_results['mean_cos_sim_real']:<15.4f} "
        f"{poisoned_results['mean_cos_sim_real']:<15.4f}"
    )

    print(
        f"{'Mean cosine similarity with FAKE':<45} "
        f"{clean_results['mean_cos_sim_fake']:<15.4f} "
        f"{poisoned_results['mean_cos_sim_fake']:<15.4f}"
    )

    print(
        f"\n{'Users preferring REAL news region':<45} "
        f"{clean_results['prefer_real_count']:<15} "
        f"{poisoned_results['prefer_real_count']:<15}"
    )

    print(
        f"{'Users preferring FAKE news region':<45} "
        f"{clean_results['prefer_fake_count']:<15} "
        f"{poisoned_results['prefer_fake_count']:<15}"
    )

    # Actual scores
    print(
        f"\n{'Mean score for REAL news (actual)':<45} "
        f"{np.mean(clean_scores_real):<15.4f} "
        f"{np.mean(poisoned_scores_real):<15.4f}"
    )

    print(
        f"{'Mean score for FAKE news (actual)':<45} "
        f"{np.mean(clean_scores_fake):<15.4f} "
        f"{np.mean(poisoned_scores_fake):<15.4f}"
    )

    clean_gap = (
        clean_results["mean_alignment_real"] - clean_results["mean_alignment_fake"]
    )
    poisoned_gap = (
        poisoned_results["mean_alignment_real"]
        - poisoned_results["mean_alignment_fake"]
    )

    clean_score_gap = np.mean(clean_scores_real) - np.mean(clean_scores_fake)
    poisoned_score_gap = np.mean(poisoned_scores_real) - np.mean(poisoned_scores_fake)

    print(
        f"\n{'Alignment gap (Real - Fake)':<45} "
        f"{clean_gap:<15.4f} "
        f"{poisoned_gap:<15.4f}"
    )

    print(
        f"{'Score gap (Real - Fake) [GROUND TRUTH]':<45} "
        f"{clean_score_gap:<15.4f} "
        f"{poisoned_score_gap:<15.4f}"
    )

    # KEY FINDINGS
    print("\n" + "=" * 75)
    print("KEY FINDINGS:")
    print("=" * 75)

    print(f"\nClean Model - Preference Gap (Real - Fake): {clean_gap:.4f}")
    print(f"Poisoned Model - Preference Gap (Real - Fake): {poisoned_gap:.4f}")
    print(
        f"\nClean Model - Score Gap (Real - Fake) [GROUND TRUTH]: {clean_score_gap:.4f}"
    )
    print(
        f"Poisoned Model - Score Gap (Real - Fake) [GROUND TRUTH]: {poisoned_score_gap:.4f}"
    )

    # Alignment-based assessment
    if poisoned_gap < 0:
        print("\n⚠️  [ALERT] Users now align MORE with FAKE news region!")
    elif poisoned_gap < clean_gap * 0.5:
        print("\n⚠️  [WARNING] Preference gap reduced significantly (>50%)")
    elif poisoned_gap < clean_gap:
        print("\n⚠️  [CAUTION] Preference for real news weakened, but still positive")
    else:
        print("\n✓ [OK] Users still strongly prefer real news region")

    # Actual score-based assessment (GROUND TRUTH)
    print("\n" + "-" * 75)
    print("GROUND TRUTH ASSESSMENT (Based on Actual Scores):")
    print("-" * 75)

    if poisoned_score_gap < 0:
        print("\n🚨 !! ATTACK SUCCESSFUL: Fake news scores HIGHER than real news!")
        print("   The model is now recommending fake news over real news.")
    elif poisoned_score_gap < clean_score_gap * 0.5:
        print("\n⚠️  !! ATTACK PARTIALLY SUCCESSFUL: Score gap reduced by >50%")
        print("   Fake news is getting significantly higher scores.")
    elif poisoned_score_gap < clean_score_gap * 0.8:
        print("\n⚠️  ! ATTACK SHOWING IMPACT: Score gap reduced by >20%")
        print("   Some effect on recommendations detected.")
    else:
        print("\n✓ [OK] Attack failed or minimal impact")
        print("    Real news still strongly preferred in recommendations.")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare representations between clean and poisoned models"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (with both model_checkpoint and poisoned_model_checkpoint)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of samples to analyze (default: all). Use smaller number for faster testing.",
    )
    args = parser.parse_args()

    # Load configuration
    print("\nLoading configuration...")
    config = load_config(args.config)

    # Validate that both checkpoints are present
    if "model_checkpoint" not in config:
        raise ValueError("Config must contain 'model_checkpoint' for clean model")
    if "poisoned_model_checkpoint" not in config:
        raise ValueError(
            "Config must contain 'poisoned_model_checkpoint' for poisoned model"
        )

    # Setup output directory from config
    output_dir = Path(config.get("output_dir", "outputs/comparison"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("EMBEDDING SPACE COMPARISON: CLEAN VS POISONED MODELS")
    print("=" * 75)
    print(f"\nClean model: {config['model_checkpoint']}")
    print(f"Poisoned model: {config['poisoned_model_checkpoint']}")
    print(f"Output directory: {output_dir}")
    if args.n_samples is not None:
        print(f"Sample limit: {args.n_samples} (for faster testing)")
    else:
        print(f"Sample limit: None (analyzing all data)")

    # Load model config (same for both models)
    model_config = load_model_config(config.get("model_config"))

    # Load test data (same for both models)
    print(f"\nLoading test data from {config['data_path']}")
    dataset, data_loader = load_test_data(config, model_config=model_config)
    print("DONE Loading data!")

    # Get data statistics (fast version)
    get_data_statistics_fast(dataset)

    # Create limited dataloader if n_samples specified
    if args.n_samples is not None:
        data_loader = create_limited_dataloader(dataset, data_loader, args.n_samples)

    # Extract representations from clean model
    print("\n" + "=" * 75)
    print("EXTRACTING CLEAN MODEL REPRESENTATIONS")
    print("=" * 75)
    clean_config = config.copy()
    clean_representations = extract_news_representations(
        data_loader, clean_config, model_config
    )

    # Extract representations from poisoned model
    print("\n" + "=" * 75)
    print("EXTRACTING POISONED MODEL REPRESENTATIONS")
    print("=" * 75)
    poisoned_config = config.copy()
    poisoned_config["model_checkpoint"] = config["poisoned_model_checkpoint"]
    poisoned_representations = extract_news_representations(
        data_loader, poisoned_config, model_config
    )

    # Create comparison subdirectories
    news_viz_dir = output_dir / "news_comparison"
    news_viz_dir.mkdir(parents=True, exist_ok=True)

    user_viz_dir = output_dir / "user_comparison"
    user_viz_dir.mkdir(parents=True, exist_ok=True)

    # Process news embeddings
    print("\n" + "=" * 75)
    print("NEWS EMBEDDINGS ANALYSIS")
    print("=" * 75)

    clean_news_emb, clean_news_labels = extract_deduplicated_embeddings(
        clean_representations["embeddings"],
        clean_representations["is_fake"],
        clean_representations.get("candidate_ids", None),
    )

    poisoned_news_emb, poisoned_news_labels = extract_deduplicated_embeddings(
        poisoned_representations["embeddings"],
        poisoned_representations["is_fake"],
        poisoned_representations.get("candidate_ids", None),
    )

    print(f"\nClean model - News embeddings shape: {clean_news_emb.shape}")
    print(f"Poisoned model - News embeddings shape: {poisoned_news_emb.shape}")

    # Compute statistics
    print("\nComputing statistics...")
    clean_news_stats = compute_embedding_stats(
        clean_news_emb, clean_news_labels, "Clean News"
    )
    poisoned_news_stats = compute_embedding_stats(
        poisoned_news_emb, poisoned_news_labels, "Poisoned News"
    )

    # Create visualizations
    print("\nCreating visualizations...")
    print("  - Norm distributions...")
    plot_embedding_norms(
        clean_news_emb,
        poisoned_news_emb,
        clean_news_labels,
        poisoned_news_labels,
        news_viz_dir / "news_norms.png",
        "News Embeddings",
    )

    print("  - PCA comparison...")
    plot_pca_comparison(
        clean_news_emb,
        poisoned_news_emb,
        clean_news_labels,
        poisoned_news_labels,
        news_viz_dir / "news_pca.png",
        "News Embeddings",
    )

    print("  - t-SNE comparison...")
    plot_tsne_comparison(
        clean_news_emb,
        poisoned_news_emb,
        clean_news_labels,
        poisoned_news_labels,
        news_viz_dir / "news_tsne.png",
        "News Embeddings",
    )

    print("  - Dimension statistics...")
    plot_dimension_stats(
        clean_news_emb,
        poisoned_news_emb,
        clean_news_labels,
        poisoned_news_labels,
        news_viz_dir / "news_dimensions.png",
        "News Embeddings",
    )

    print("  - Separation metrics...")
    plot_separation_metrics(
        clean_news_stats,
        poisoned_news_stats,
        "News",
        news_viz_dir / "news_separation_metrics.png",
    )

    # Print comparison report
    print_comparison_report(clean_news_stats, poisoned_news_stats, "News")

    # Process user embeddings
    print("\n" + "=" * 75)
    print("USER EMBEDDINGS ANALYSIS")
    print("=" * 75)

    clean_user_emb, clean_user_labels = extract_deduplicated_embeddings(
        clean_representations["user_embeddings"],
        clean_representations["is_fake"][:, 0],  # Take first candidate's label
        clean_representations.get("user_ids", None),
    )

    poisoned_user_emb, poisoned_user_labels = extract_deduplicated_embeddings(
        poisoned_representations["user_embeddings"],
        poisoned_representations["is_fake"][:, 0],
        poisoned_representations.get("user_ids", None),
    )

    print(f"\nClean model - User embeddings shape: {clean_user_emb.shape}")
    print(f"Poisoned model - User embeddings shape: {poisoned_user_emb.shape}")

    # Compute statistics
    print("\nComputing statistics...")
    clean_user_stats = compute_embedding_stats(
        clean_user_emb, clean_user_labels, "Clean User"
    )
    poisoned_user_stats = compute_embedding_stats(
        poisoned_user_emb, poisoned_user_labels, "Poisoned User"
    )

    # Create visualizations
    print("\nCreating visualizations...")
    print("  - Norm distributions...")
    plot_embedding_norms(
        clean_user_emb,
        poisoned_user_emb,
        clean_user_labels,
        poisoned_user_labels,
        user_viz_dir / "user_norms.png",
        "User Embeddings",
    )

    print("  - PCA comparison...")
    plot_pca_comparison(
        clean_user_emb,
        poisoned_user_emb,
        clean_user_labels,
        poisoned_user_labels,
        user_viz_dir / "user_pca.png",
        "User Embeddings",
    )

    print("  - t-SNE comparison...")
    plot_tsne_comparison(
        clean_user_emb,
        poisoned_user_emb,
        clean_user_labels,
        poisoned_user_labels,
        user_viz_dir / "user_tsne.png",
        "User Embeddings",
    )

    print("  - Dimension statistics...")
    plot_dimension_stats(
        clean_user_emb,
        poisoned_user_emb,
        clean_user_labels,
        poisoned_user_labels,
        user_viz_dir / "user_dimensions.png",
        "User Embeddings",
    )

    print("  - Separation metrics...")
    plot_separation_metrics(
        clean_user_stats,
        poisoned_user_stats,
        "User",
        user_viz_dir / "user_separation_metrics.png",
    )

    # Print comparison report
    print_comparison_report(clean_user_stats, poisoned_user_stats, "User")

    # USER PREFERENCE ANALYSIS
    print("\n" + "=" * 75)
    print("ANALYZING USER PREFERENCES")
    print("=" * 75)

    # Create preference analysis subdirectory
    preference_viz_dir = output_dir / "preference_analysis"
    preference_viz_dir.mkdir(parents=True, exist_ok=True)

    # Method 1: User-Centroid Alignment Analysis
    print("\nComputing user-centroid alignment...")
    clean_pref_results = analyze_user_preferences(
        clean_user_emb, clean_news_emb, clean_news_labels, "Clean"
    )
    poisoned_pref_results = analyze_user_preferences(
        poisoned_user_emb, poisoned_news_emb, poisoned_news_labels, "Poisoned"
    )

    # Method 2: Visualize User-Centroid Alignment
    print("Creating user-centroid alignment visualization...")
    plot_user_centroid_alignment(
        clean_pref_results,
        poisoned_pref_results,
        preference_viz_dir / "user_centroid_alignment.png",
    )

    # Method 3: Compute Actual Recommendation Scores (Ground Truth)
    print("Computing actual recommendation scores...")
    clean_scores_real, clean_scores_fake = compute_actual_scores(
        clean_user_emb, clean_news_emb, clean_news_labels
    )
    poisoned_scores_real, poisoned_scores_fake = compute_actual_scores(
        poisoned_user_emb, poisoned_news_emb, poisoned_news_labels
    )

    # Visualize actual scores
    print("Creating actual recommendation scores visualization...")
    plot_actual_recommendation_scores(
        clean_scores_real,
        clean_scores_fake,
        poisoned_scores_real,
        poisoned_scores_fake,
        preference_viz_dir / "actual_recommendation_scores.png",
    )

    # Print comprehensive preference analysis report
    print_user_preference_analysis(
        clean_pref_results,
        poisoned_pref_results,
        clean_scores_real,
        clean_scores_fake,
        poisoned_scores_real,
        poisoned_scores_fake,
    )

    # Save statistics to JSON
    print("\nSaving statistics...")

    # Remove non-serializable items from preference results for JSON
    clean_pref_json = {
        k: v
        for k, v in clean_pref_results.items()
        if k
        not in [
            "alignment_real_all",
            "alignment_fake_all",
            "real_centroid",
            "fake_centroid",
            "mean_user",
        ]
    }
    poisoned_pref_json = {
        k: v
        for k, v in poisoned_pref_results.items()
        if k
        not in [
            "alignment_real_all",
            "alignment_fake_all",
            "real_centroid",
            "fake_centroid",
            "mean_user",
        ]
    }

    stats = {
        "config": {
            "n_samples_analyzed": args.n_samples if args.n_samples else "all",
            "clean_model": config["model_checkpoint"],
            "poisoned_model": config["poisoned_model_checkpoint"],
        },
        "news": {"clean": clean_news_stats, "poisoned": poisoned_news_stats},
        "user": {"clean": clean_user_stats, "poisoned": poisoned_user_stats},
        "preference_analysis": {
            "clean": clean_pref_json,
            "poisoned": poisoned_pref_json,
            "actual_scores": {
                "clean_real_mean": float(np.mean(clean_scores_real)),
                "clean_fake_mean": float(np.mean(clean_scores_fake)),
                "poisoned_real_mean": float(np.mean(poisoned_scores_real)),
                "poisoned_fake_mean": float(np.mean(poisoned_scores_fake)),
                "clean_gap": float(
                    np.mean(clean_scores_real) - np.mean(clean_scores_fake)
                ),
                "poisoned_gap": float(
                    np.mean(poisoned_scores_real) - np.mean(poisoned_scores_fake)
                ),
            },
        },
    }

    stats_path = output_dir / "comparison_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_path}")

    # Summary
    print("\n" + "=" * 75)
    print("ANALYSIS COMPLETE")
    print("=" * 75)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - News comparison: {news_viz_dir}")
    print(f"  - User comparison: {user_viz_dir}")
    print(f"  - Preference analysis: {preference_viz_dir}")
    print(f"  - Statistics: {stats_path}")

    if args.n_samples is not None:
        print(f"\nNote: Analysis performed on {args.n_samples} samples.")
        print(f"Run without --n_samples to analyze all data.")

    print()


if __name__ == "__main__":
    main()
