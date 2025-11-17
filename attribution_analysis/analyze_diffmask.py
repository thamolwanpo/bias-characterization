"""
DIFFMASK Attribution Analysis for News Recommendation Models.

This script trains a DIFFMASK interpreter network and uses it to analyze
which words affect model recommendations for real vs fake news classification.

Reference: Learning to Faithfully Rationalize by Construction (de Cao et al., 2020)
https://arxiv.org/abs/2005.00115
"""

import argparse
import yaml
import torch
import json
import os
import sys
from datetime import datetime

# Add parent directory for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import from representation_analysis
sys.path.insert(0, os.path.join(parent_dir, "representation_analysis"))
from representation import load_model

# Import DIFFMASK components
from methods.diffmask import DIFFMASK
from methods.diffmask.train_diffmask import (
    train_diffmask,
    extract_attributions_diffmask,
    load_diffmask_checkpoint,
)

# Import existing attribution analysis functions
from attribution import (
    analyze_word_importance,
    plot_word_importance,
    plot_attribution_heatmap,
    compare_attributions,
)

# Import data loader
from data_loader import create_dataloaders


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DIFFMASK attribution analysis for news recommendation models"
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file (YAML)"
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples to analyze (default: 100)",
    )

    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of training epochs for DIFFMASK (default: 10)",
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=768,
        help="Hidden dimension of model (default: 768 for BERT)",
    )

    parser.add_argument(
        "--num_probe_layers",
        type=int,
        default=3,
        help="Number of layers to probe (default: 3)",
    )

    parser.add_argument(
        "--probe_type",
        type=str,
        default="simple",
        choices=["simple", "layerwise"],
        help="Type of probe network (default: simple)",
    )

    parser.add_argument(
        "--constraint_margin",
        type=float,
        default=0.1,
        help="Constraint margin for divergence (default: 0.1)",
    )

    parser.add_argument(
        "--lr_probe",
        type=float,
        default=1e-3,
        help="Learning rate for probe network (default: 1e-3)",
    )

    parser.add_argument(
        "--lr_baseline",
        type=float,
        default=1e-3,
        help="Learning rate for baseline vector (default: 1e-3)",
    )

    parser.add_argument(
        "--lr_lambda",
        type=float,
        default=1e-2,
        help="Learning rate for Lagrangian multiplier (default: 1e-2)",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=15,
        help="Number of top words to display in plots (default: 15)",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to DIFFMASK checkpoint to load (optional, will train if not provided)",
    )

    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and only do inference (requires --checkpoint)",
    )

    parser.add_argument(
        "--train_dataset",
        type=str,
        default="train_clean",
        choices=["train_clean", "train_poisoned", "benchmark"],
        help="Dataset for training DIFFMASK interpreter (default: train_clean)",
    )

    parser.add_argument(
        "--test_dataset",
        type=str,
        default="benchmark",
        choices=["train_clean", "train_poisoned", "benchmark"],
        help="Dataset for extracting attributions (default: benchmark)",
    )

    return parser.parse_args()


def main():
    """Main analysis pipeline."""
    args = parse_args()

    # Load configuration
    print(f"Loading config from: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load model config
    print(f"Loading model config from: {config['model_config']}")
    with open(config["model_config"], "r") as f:
        model_config_dict = yaml.safe_load(f)

    # Convert to object with attributes
    class ModelConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)

    model_config = ModelConfig(model_config_dict)

    # Set device
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = config.get("output_dir", "./output/diffmask")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Output directory: {run_dir}")

    # Create data loaders
    print("\nCreating data loaders...")
    print(f"  Training dataset: {args.train_dataset}")
    print(f"  Test dataset: {args.test_dataset}")

    # Load training data (for training DIFFMASK interpreter)
    _, train_loader = create_dataloaders(config, model_config, dataset_type=args.train_dataset)

    # Load test data (for extracting attributions)
    _, test_loader = create_dataloaders(config, model_config, dataset_type=args.test_dataset)

    # Load model
    print(f"\nLoading model from: {config['model_checkpoint']}")
    lit_model = load_model(config, model_config)
    model = lit_model.model
    model.eval()
    model = model.to(device)

    # Initialize DIFFMASK
    print("\nInitializing DIFFMASK...")
    diffmask_clean = DIFFMASK(
        model=model,
        hidden_dim=args.hidden_dim,
        num_probe_layers=args.num_probe_layers,
        probe_type=args.probe_type,
        constraint_margin=args.constraint_margin,
    )

    # Training or loading checkpoint
    if args.skip_training and args.checkpoint:
        print(f"\nLoading DIFFMASK checkpoint from: {args.checkpoint}")
        diffmask_clean = load_diffmask_checkpoint(
            diffmask_clean, args.checkpoint, device
        )
    elif args.checkpoint:
        print(f"\nLoading DIFFMASK checkpoint from: {args.checkpoint}")
        diffmask_clean = load_diffmask_checkpoint(
            diffmask_clean, args.checkpoint, device
        )
    else:
        print(f"\nTraining DIFFMASK for {args.n_epochs} epochs...")
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        history = train_diffmask(
            diffmask_clean,
            train_loader,
            n_epochs=args.n_epochs,
            lr_probe=args.lr_probe,
            lr_baseline=args.lr_baseline,
            lr_lambda=args.lr_lambda,
            device=device,
            save_dir=checkpoint_dir,
            verbose=True,
        )

        # Save training history
        history_path = os.path.join(run_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nSaved training history to: {history_path}")

    # Extract attributions for clean model
    print("\n" + "=" * 80)
    print("Extracting attributions for CLEAN model")
    print("=" * 80)
    attributions_clean = extract_attributions_diffmask(
        diffmask_clean,
        test_loader,
        n_samples=args.n_samples,
        device=device,
        verbose=True,
    )

    # Save attributions
    save_path_clean = os.path.join(run_dir, "attributions_clean.json")
    with open(save_path_clean, "w") as f:
        json.dump(
            {
                "attributions": [
                    attr.tolist() for attr in attributions_clean["attributions"]
                ],
                "tokens": attributions_clean["tokens"],
                "labels": attributions_clean["labels"].tolist(),
                "scores": attributions_clean["scores"].tolist(),
                "predictions": attributions_clean["predictions"].tolist(),
            },
            f,
            indent=2,
        )
    print(f"\nSaved clean model attributions to: {save_path_clean}")

    # Analyze poisoned model (if available)
    if "poisoned_model_checkpoint" in config:
        print("\n" + "=" * 80)
        print("Extracting attributions for POISONED model")
        print("=" * 80)

        # Load poisoned model
        config_poisoned = config.copy()
        config_poisoned["model_checkpoint"] = config["poisoned_model_checkpoint"]

        lit_model_poisoned = load_model(config_poisoned, model_config)
        model_poisoned = lit_model_poisoned.model
        model_poisoned.eval()
        model_poisoned = model_poisoned.to(device)

        # Load training data for poisoned model
        # Poisoned model was trained on train_poisoned, so train its interpreter on train_poisoned
        print("\nLoading training data for poisoned model interpreter...")
        if args.train_dataset == "train_clean":
            # If user specified train_clean for clean model, use train_poisoned for poisoned model
            train_dataset_poisoned = "train_poisoned"
        else:
            # Otherwise use the same dataset (though this might not make sense)
            train_dataset_poisoned = args.train_dataset

        print(f"  Training dataset for poisoned model: {train_dataset_poisoned}")
        _, train_loader_poisoned = create_dataloaders(config, model_config, dataset_type=train_dataset_poisoned)

        # Initialize DIFFMASK for poisoned model
        diffmask_poisoned = DIFFMASK(
            model=model_poisoned,
            hidden_dim=args.hidden_dim,
            num_probe_layers=args.num_probe_layers,
            probe_type=args.probe_type,
            constraint_margin=args.constraint_margin,
        )

        # Train DIFFMASK for poisoned model
        print(f"\nTraining DIFFMASK for poisoned model ({args.n_epochs} epochs)...")
        checkpoint_dir_poisoned = os.path.join(run_dir, "checkpoints_poisoned")
        history_poisoned = train_diffmask(
            diffmask_poisoned,
            train_loader_poisoned,  # Use poisoned training data
            n_epochs=args.n_epochs,
            lr_probe=args.lr_probe,
            lr_baseline=args.lr_baseline,
            lr_lambda=args.lr_lambda,
            device=device,
            save_dir=checkpoint_dir_poisoned,
            verbose=True,
        )

        # Extract attributions
        attributions_poisoned = extract_attributions_diffmask(
            diffmask_poisoned,
            test_loader,
            n_samples=args.n_samples,
            device=device,
            verbose=True,
        )

        # Save attributions
        save_path_poisoned = os.path.join(run_dir, "attributions_poisoned.json")
        with open(save_path_poisoned, "w") as f:
            json.dump(
                {
                    "attributions": [
                        attr.tolist() for attr in attributions_poisoned["attributions"]
                    ],
                    "tokens": attributions_poisoned["tokens"],
                    "labels": attributions_poisoned["labels"].tolist(),
                    "scores": attributions_poisoned["scores"].tolist(),
                    "predictions": attributions_poisoned["predictions"].tolist(),
                },
                f,
                indent=2,
            )
        print(f"\nSaved poisoned model attributions to: {save_path_poisoned}")

    else:
        attributions_poisoned = None

    # Analyze word importance
    print("\n" + "=" * 80)
    print("Analyzing word importance")
    print("=" * 80)

    if attributions_poisoned:
        word_importance = analyze_word_importance(
            attributions_clean, attributions_poisoned, top_k=args.top_k
        )
    else:
        # Create dummy poisoned data for single-model analysis
        word_importance = {
            "clean": analyze_word_importance(
                attributions_clean,
                attributions_clean,  # Use same data
                top_k=args.top_k,
            )["clean"]
        }

    # Save word importance
    importance_path = os.path.join(run_dir, "word_importance.json")
    with open(importance_path, "w") as f:
        # Convert defaultdict to regular dict for JSON serialization
        importance_serializable = {}
        for model_key in word_importance:
            importance_serializable[model_key] = {}
            for label_key in word_importance[model_key]:
                importance_serializable[model_key][label_key] = dict(
                    word_importance[model_key][label_key]
                )
        json.dump(importance_serializable, f, indent=2)
    print(f"Saved word importance to: {importance_path}")

    # Create visualizations
    print("\n" + "=" * 80)
    print("Creating visualizations")
    print("=" * 80)

    # Plot word importance
    plot_path = os.path.join(run_dir, "word_importance.png")
    plot_word_importance(word_importance, plot_path, top_k=args.top_k)

    # Plot attribution heatmaps
    heatmap_path_clean = os.path.join(run_dir, "attribution_heatmap_clean.png")
    plot_attribution_heatmap(attributions_clean, heatmap_path_clean, n_samples=10)

    if attributions_poisoned:
        heatmap_path_poisoned = os.path.join(
            run_dir, "attribution_heatmap_poisoned.png"
        )
        plot_attribution_heatmap(
            attributions_poisoned, heatmap_path_poisoned, n_samples=10
        )

        # Compare attributions
        comparison_path = os.path.join(run_dir, "attribution_comparison.png")
        compare_attributions(
            {"clean": word_importance["clean"]},
            {"poisoned": word_importance["poisoned"]},
            comparison_path,
            top_k=args.top_k,
        )

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
