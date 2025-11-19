"""
Main script for TRAK analysis on news recommendation models.

Computes TRAK scores for training samples and analyzes their influence.
"""

import argparse
import yaml
from pathlib import Path
import torch
import numpy as np
import sys
import os

# Add plm4newsrs to path
sys.path.insert(
    0,
    os.path.abspath("/home/thamo/PhD/bias-characterization/plm4newsrs/src"),
)

from data_loader import load_train_data, get_data_statistics_fast
from trak_scorer import TRAKScorer, compute_trak_scores, rank_training_samples
from visualize import (
    plot_score_distribution,
    plot_top_influential_samples,
    plot_score_comparison,
    save_ranking_to_csv,
)


def load_model(model_checkpoint_path, model_config_path, device='cuda'):
    """
    Load a trained model from checkpoint.

    Args:
        model_checkpoint_path: Path to model checkpoint
        model_config_path: Path to model config YAML
        device: Device to load model on

    Returns:
        Loaded model
    """
    from models import load_plm_model
    from omegaconf import OmegaConf

    print(f"\nLoading model from: {model_checkpoint_path}")
    print(f"Using config: {model_config_path}")

    # Load model config
    model_config = OmegaConf.load(model_config_path)

    # Load model
    model = load_plm_model(model_config)

    # Load checkpoint
    checkpoint = torch.load(model_checkpoint_path, map_location=device)

    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'model.' prefix if present (from Lightning)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('model.', '') if k.startswith('model.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    print("Model loaded successfully!")

    return model, model_config


def main():
    parser = argparse.ArgumentParser(description="TRAK Analysis for News Recommendation")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file',
    )
    parser.add_argument(
        '--dataset_type',
        type=str,
        default='train_clean',
        choices=['train_clean', 'train_poisoned'],
        help='Dataset type to analyze',
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='clean',
        choices=['clean', 'poisoned'],
        help='Model type to use for analysis',
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=100,
        help='Number of top samples to analyze',
    )
    parser.add_argument(
        '--proj_dim',
        type=int,
        default=512,
        help='Projection dimension for TRAK',
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set up output directory
    output_dir = Path(config['output_dir']) / f"{args.model_type}_{args.dataset_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Load model
    if args.model_type == 'clean':
        model_checkpoint = config['model_checkpoint']
    else:
        model_checkpoint = config['poisoned_model_checkpoint']

    model, model_config = load_model(
        model_checkpoint_path=model_checkpoint,
        model_config_path=config['model_config'],
        device=config.get('device', 'cuda'),
    )

    # Load training data
    print("\n" + "="*80)
    print(f"Loading {args.dataset_type} dataset")
    print("="*80)

    dataset, dataloader = load_train_data(
        config=config,
        model_config=model_config,
        dataset_type=args.dataset_type,
    )

    # Get data statistics
    stats = get_data_statistics_fast(dataset)

    # Compute TRAK scores
    print("\n" + "="*80)
    print("Computing TRAK Scores")
    print("="*80)

    results = compute_trak_scores(
        model=model,
        train_loader=dataloader,
        device=config.get('device', 'cuda'),
        proj_dim=args.proj_dim,
        save_dir=output_dir,
    )

    # Rank training samples
    print("\n" + "="*80)
    print("Ranking Training Samples")
    print("="*80)

    ranking_results = rank_training_samples(
        scores=results['scores'],
        labels=results['labels'],
        top_k=args.top_k,
    )

    # Visualizations
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)

    # Score distribution
    plot_score_distribution(
        scores=results['scores'],
        labels=results['labels'],
        save_path=output_dir / 'score_distribution.png',
        title=f"TRAK Scores - {args.model_type.title()} Model on {args.dataset_type}",
    )

    # Top influential samples
    plot_top_influential_samples(
        ranking_results=ranking_results,
        save_path=output_dir / 'top_influential_samples.png',
        top_k=50,
    )

    # Save ranking to CSV
    save_ranking_to_csv(
        ranking_results=ranking_results,
        dataset=dataset,
        save_path=output_dir / 'ranking_results.csv',
        top_k=args.top_k,
    )

    # Save metadata
    metadata = {
        'config': config,
        'dataset_type': args.dataset_type,
        'model_type': args.model_type,
        'top_k': args.top_k,
        'proj_dim': args.proj_dim,
        'data_stats': stats,
        'score_stats': {
            'mean': float(results['scores'].mean()),
            'std': float(results['scores'].std()),
            'min': float(results['scores'].min()),
            'max': float(results['scores'].max()),
        },
    }

    import json
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - trak_scores.npy")
    print(f"  - trak_labels.npy")
    print(f"  - trak_gradients.npy")
    print(f"  - ranking_results.csv")
    print(f"  - score_distribution.png")
    print(f"  - top_influential_samples.png")
    print(f"  - metadata.json")


if __name__ == '__main__':
    main()
