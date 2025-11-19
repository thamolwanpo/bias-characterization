"""
Compare TRAK scores between clean and poisoned models.

This script loads pre-computed TRAK scores from both models and generates
comparison visualizations.
"""

import argparse
import numpy as np
from pathlib import Path
import json

from visualize import plot_score_comparison


def main():
    parser = argparse.ArgumentParser(description="Compare TRAK scores between models")
    parser.add_argument(
        '--clean_dir',
        type=str,
        required=True,
        help='Directory with clean model results',
    )
    parser.add_argument(
        '--poisoned_dir',
        type=str,
        required=True,
        help='Directory with poisoned model results',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./trak_comparison',
        help='Output directory for comparison results',
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("Loading TRAK Results")
    print("="*80)

    # Load clean model results
    clean_dir = Path(args.clean_dir)
    clean_scores = np.load(clean_dir / 'trak_scores.npy')
    clean_labels = np.load(clean_dir / 'trak_labels.npy')

    print(f"\nClean model results from: {clean_dir}")
    print(f"  Samples: {len(clean_scores)}")
    print(f"  Fake: {np.sum(clean_labels == 1)}")
    print(f"  Real: {np.sum(clean_labels == 0)}")

    # Load poisoned model results
    poisoned_dir = Path(args.poisoned_dir)
    poisoned_scores = np.load(poisoned_dir / 'trak_scores.npy')
    poisoned_labels = np.load(poisoned_dir / 'trak_labels.npy')

    print(f"\nPoisoned model results from: {poisoned_dir}")
    print(f"  Samples: {len(poisoned_scores)}")
    print(f"  Fake: {np.sum(poisoned_labels == 1)}")
    print(f"  Real: {np.sum(poisoned_labels == 0)}")

    # Generate comparison plot
    print("\n" + "="*80)
    print("Generating Comparison Visualizations")
    print("="*80)

    plot_score_comparison(
        clean_scores=clean_scores,
        clean_labels=clean_labels,
        poisoned_scores=poisoned_scores,
        poisoned_labels=poisoned_labels,
        save_path=output_dir / 'model_comparison.png',
    )

    # Compute additional statistics
    print("\n" + "="*80)
    print("Computing Comparison Statistics")
    print("="*80)

    min_len = min(len(clean_scores), len(poisoned_scores))
    score_diff = poisoned_scores[:min_len] - clean_scores[:min_len]

    # Find samples with largest score changes
    largest_increases = np.argsort(score_diff)[::-1][:100]
    largest_decreases = np.argsort(score_diff)[:100]

    print(f"\nTop 10 samples with largest score increases (poisoned - clean):")
    for i, idx in enumerate(largest_increases[:10]):
        label = 'fake' if clean_labels[idx] == 1 else 'real'
        print(f"  {i+1}. Sample {idx} ({label}): "
              f"clean={clean_scores[idx]:.4f}, "
              f"poisoned={poisoned_scores[idx]:.4f}, "
              f"diff={score_diff[idx]:.4f}")

    print(f"\nTop 10 samples with largest score decreases (poisoned - clean):")
    for i, idx in enumerate(largest_decreases[:10]):
        label = 'fake' if clean_labels[idx] == 1 else 'real'
        print(f"  {i+1}. Sample {idx} ({label}): "
              f"clean={clean_scores[idx]:.4f}, "
              f"poisoned={poisoned_scores[idx]:.4f}, "
              f"diff={score_diff[idx]:.4f}")

    # Save comparison statistics
    comparison_stats = {
        'clean_model': {
            'n_samples': len(clean_scores),
            'n_fake': int(np.sum(clean_labels == 1)),
            'n_real': int(np.sum(clean_labels == 0)),
            'score_mean': float(clean_scores.mean()),
            'score_std': float(clean_scores.std()),
        },
        'poisoned_model': {
            'n_samples': len(poisoned_scores),
            'n_fake': int(np.sum(poisoned_labels == 1)),
            'n_real': int(np.sum(poisoned_labels == 0)),
            'score_mean': float(poisoned_scores.mean()),
            'score_std': float(poisoned_scores.std()),
        },
        'comparison': {
            'score_diff_mean': float(score_diff.mean()),
            'score_diff_std': float(score_diff.std()),
            'correlation': float(np.corrcoef(clean_scores[:min_len], poisoned_scores[:min_len])[0, 1]),
            'largest_increases': [
                {
                    'sample_idx': int(idx),
                    'label': 'fake' if clean_labels[idx] == 1 else 'real',
                    'clean_score': float(clean_scores[idx]),
                    'poisoned_score': float(poisoned_scores[idx]),
                    'diff': float(score_diff[idx]),
                }
                for idx in largest_increases[:100]
            ],
            'largest_decreases': [
                {
                    'sample_idx': int(idx),
                    'label': 'fake' if clean_labels[idx] == 1 else 'real',
                    'clean_score': float(clean_scores[idx]),
                    'poisoned_score': float(poisoned_scores[idx]),
                    'diff': float(score_diff[idx]),
                }
                for idx in largest_decreases[:100]
            ],
        },
    }

    with open(output_dir / 'comparison_stats.json', 'w') as f:
        json.dump(comparison_stats, f, indent=2)

    print("\n" + "="*80)
    print("Comparison Complete!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - model_comparison.png")
    print(f"  - comparison_stats.json")


if __name__ == '__main__':
    main()
