"""
Visualization tools for TRAK analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_score_distribution(
    scores,
    labels,
    save_path=None,
    title="TRAK Score Distribution",
):
    """
    Plot distribution of TRAK scores.

    Args:
        scores: TRAK scores
        labels: Labels (0=real, 1=fake)
        save_path: Path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall distribution
    axes[0].hist(scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(scores.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0].set_xlabel('TRAK Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'{title} (All Samples)', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Distribution by class
    fake_scores = scores[labels == 1]
    real_scores = scores[labels == 0]

    axes[1].hist(
        fake_scores,
        bins=50,
        alpha=0.6,
        color='red',
        label=f'Fake News (n={len(fake_scores)})',
        edgecolor='black',
    )
    axes[1].hist(
        real_scores,
        bins=50,
        alpha=0.6,
        color='green',
        label=f'Real News (n={len(real_scores)})',
        edgecolor='black',
    )
    axes[1].set_xlabel('TRAK Score', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'{title} by Class', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved score distribution plot to: {save_path}")
    else:
        plt.show()

    plt.close()

    # Print statistics
    print("\n" + "="*80)
    print("Score Distribution Statistics")
    print("="*80)
    print(f"\nAll samples:")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Std: {scores.std():.4f}")
    print(f"  Min: {scores.min():.4f}")
    print(f"  Max: {scores.max():.4f}")

    print(f"\nFake news (n={len(fake_scores)}):")
    print(f"  Mean: {fake_scores.mean():.4f}")
    print(f"  Std: {fake_scores.std():.4f}")
    print(f"  Min: {fake_scores.min():.4f}")
    print(f"  Max: {fake_scores.max():.4f}")

    print(f"\nReal news (n={len(real_scores)}):")
    print(f"  Mean: {real_scores.mean():.4f}")
    print(f"  Std: {real_scores.std():.4f}")
    print(f"  Min: {real_scores.min():.4f}")
    print(f"  Max: {real_scores.max():.4f}")


def plot_top_influential_samples(
    ranking_results,
    save_path=None,
    top_k=50,
):
    """
    Plot top influential samples.

    Args:
        ranking_results: Results from rank_training_samples()
        save_path: Path to save figure
        top_k: Number of top samples to show
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-k overall
    top_scores = ranking_results['top_k_scores'][:top_k]
    top_labels = ranking_results['top_k_labels'][:top_k]
    colors = ['red' if label == 1 else 'green' for label in top_labels]

    axes[0, 0].barh(range(top_k), top_scores[::-1], color=colors[::-1], alpha=0.7)
    axes[0, 0].set_xlabel('TRAK Score', fontsize=12)
    axes[0, 0].set_ylabel('Sample Rank', fontsize=12)
    axes[0, 0].set_title(f'Top-{top_k} Most Influential Samples', fontsize=14)
    axes[0, 0].grid(alpha=0.3, axis='x')

    # Bottom-k overall
    bottom_scores = ranking_results['bottom_k_scores'][:top_k]
    bottom_labels = ranking_results['bottom_k_labels'][:top_k]
    colors_bottom = ['red' if label == 1 else 'green' for label in bottom_labels]

    axes[0, 1].barh(range(top_k), bottom_scores[::-1], color=colors_bottom[::-1], alpha=0.7)
    axes[0, 1].set_xlabel('TRAK Score', fontsize=12)
    axes[0, 1].set_ylabel('Sample Rank', fontsize=12)
    axes[0, 1].set_title(f'Bottom-{top_k} Least Influential Samples', fontsize=14)
    axes[0, 1].grid(alpha=0.3, axis='x')

    # Top fake news
    top_fake_scores = ranking_results['top_fake_scores'][:top_k]

    axes[1, 0].barh(range(len(top_fake_scores)), top_fake_scores[::-1], color='red', alpha=0.7)
    axes[1, 0].set_xlabel('TRAK Score', fontsize=12)
    axes[1, 0].set_ylabel('Sample Rank', fontsize=12)
    axes[1, 0].set_title(f'Top-{top_k} Fake News Samples', fontsize=14)
    axes[1, 0].grid(alpha=0.3, axis='x')

    # Top real news
    top_real_scores = ranking_results['top_real_scores'][:top_k]

    axes[1, 1].barh(range(len(top_real_scores)), top_real_scores[::-1], color='green', alpha=0.7)
    axes[1, 1].set_xlabel('TRAK Score', fontsize=12)
    axes[1, 1].set_ylabel('Sample Rank', fontsize=12)
    axes[1, 1].set_title(f'Top-{top_k} Real News Samples', fontsize=14)
    axes[1, 1].grid(alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved top influential samples plot to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_score_comparison(
    clean_scores,
    clean_labels,
    poisoned_scores,
    poisoned_labels,
    save_path=None,
):
    """
    Compare TRAK scores between clean and poisoned models.

    Args:
        clean_scores: Scores from clean model
        clean_labels: Labels from clean model
        poisoned_scores: Scores from poisoned model
        poisoned_labels: Labels from poisoned model
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Clean model distribution
    axes[0, 0].hist(
        clean_scores[clean_labels == 1],
        bins=50,
        alpha=0.6,
        color='red',
        label='Fake',
        edgecolor='black',
    )
    axes[0, 0].hist(
        clean_scores[clean_labels == 0],
        bins=50,
        alpha=0.6,
        color='green',
        label='Real',
        edgecolor='black',
    )
    axes[0, 0].set_xlabel('TRAK Score', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Clean Model - Score Distribution', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Poisoned model distribution
    axes[0, 1].hist(
        poisoned_scores[poisoned_labels == 1],
        bins=50,
        alpha=0.6,
        color='red',
        label='Fake',
        edgecolor='black',
    )
    axes[0, 1].hist(
        poisoned_scores[poisoned_labels == 0],
        bins=50,
        alpha=0.6,
        color='green',
        label='Real',
        edgecolor='black',
    )
    axes[0, 1].set_xlabel('TRAK Score', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Poisoned Model - Score Distribution', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Score comparison scatter plot
    min_len = min(len(clean_scores), len(poisoned_scores))
    axes[1, 0].scatter(
        clean_scores[:min_len],
        poisoned_scores[:min_len],
        alpha=0.5,
        s=10,
    )
    axes[1, 0].plot(
        [clean_scores.min(), clean_scores.max()],
        [clean_scores.min(), clean_scores.max()],
        'r--',
        linewidth=2,
        label='y=x',
    )
    axes[1, 0].set_xlabel('Clean Model Score', fontsize=12)
    axes[1, 0].set_ylabel('Poisoned Model Score', fontsize=12)
    axes[1, 0].set_title('Score Comparison (Clean vs Poisoned)', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Difference distribution
    score_diff = poisoned_scores[:min_len] - clean_scores[:min_len]
    axes[1, 1].hist(score_diff, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    axes[1, 1].set_xlabel('Score Difference (Poisoned - Clean)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Score Difference Distribution', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved score comparison plot to: {save_path}")
    else:
        plt.show()

    plt.close()

    # Print comparison statistics
    print("\n" + "="*80)
    print("Score Comparison Statistics")
    print("="*80)

    print("\nClean Model:")
    print(f"  Mean: {clean_scores.mean():.4f}")
    print(f"  Std: {clean_scores.std():.4f}")

    print("\nPoisoned Model:")
    print(f"  Mean: {poisoned_scores.mean():.4f}")
    print(f"  Std: {poisoned_scores.std():.4f}")

    print(f"\nScore Difference (Poisoned - Clean):")
    print(f"  Mean: {score_diff.mean():.4f}")
    print(f"  Std: {score_diff.std():.4f}")
    print(f"  Correlation: {np.corrcoef(clean_scores[:min_len], poisoned_scores[:min_len])[0, 1]:.4f}")


def save_ranking_to_csv(
    ranking_results,
    dataset,
    save_path,
    top_k=100,
):
    """
    Save ranking results to CSV for further analysis.

    Args:
        ranking_results: Results from rank_training_samples()
        dataset: Dataset object (to get sample info)
        save_path: Path to save CSV
        top_k: Number of top samples to save
    """
    import pandas as pd

    records = []

    # Top-k samples
    for i, (idx, score, label) in enumerate(zip(
        ranking_results['top_k_indices'][:top_k],
        ranking_results['top_k_scores'][:top_k],
        ranking_results['top_k_labels'][:top_k],
    )):
        records.append({
            'rank': i + 1,
            'sample_idx': int(idx),
            'score': float(score),
            'label': 'fake' if label == 1 else 'real',
            'category': 'top_influential',
        })

    # Bottom-k samples
    for i, (idx, score, label) in enumerate(zip(
        ranking_results['bottom_k_indices'][:top_k],
        ranking_results['bottom_k_scores'][:top_k],
        ranking_results['bottom_k_labels'][:top_k],
    )):
        records.append({
            'rank': i + 1,
            'sample_idx': int(idx),
            'score': float(score),
            'label': 'fake' if label == 1 else 'real',
            'category': 'bottom_influential',
        })

    df = pd.DataFrame(records)
    df.to_csv(save_path, index=False)

    print(f"\nSaved ranking results to: {save_path}")

    return df
