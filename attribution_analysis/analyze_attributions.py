"""
Main script for axiomatic attribution analysis.

Compares word-level attributions between clean and poisoned models
to understand which words drive real vs fake news recommendations.

Usage:
    # Analyze on benchmark (unseen test data)
    python analyze_attributions.py --config configs/nrms_bert_frozen.yaml --dataset benchmark --n_samples 50

    # Analyze on clean training data (to see features learned from real news only)
    python analyze_attributions.py --config configs/nrms_bert_frozen.yaml --dataset train_clean --n_samples 50

    # Analyze on poisoned training data (to see features learned from fake + real news)
    python analyze_attributions.py --config configs/nrms_bert_frozen.yaml --dataset train_poisoned --n_samples 50
"""

import argparse
import yaml
from pathlib import Path
import json
import numpy as np
import sys
import os

# Add parent directory for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

sys.path.insert(
    0,
    os.path.abspath("/home/thamo/PhD/bias-characterization/plm4newsrs"),
)

# sys.path.insert(
#     0,
#     os.path.abspath(
#         "/content/drive/MyDrive/bias-characterized/bias-characterization/plm4newsrs"
#     ),
# )

from configs import load_config as load_model_config

# Import from current directory
from data_loader import load_test_data, get_data_statistics_fast
from attribution import (
    extract_attributions_for_dataset,
    analyze_word_importance,
    plot_word_importance,
    plot_attribution_heatmap,
    compare_attributions,
    analyze_word_frequency_from_top_samples,
    plot_word_frequency_from_top_samples,
)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def print_top_words(word_importance, model_name, label, top_k=10):
    """Print top attributed words for a model and label."""
    words_data = word_importance[model_name][label]

    if not words_data:
        print(f"  No data for {model_name} - {label}")
        return

    # Get top_k positive words (highest positive mean attributions)
    positive_words = [(word, stats) for word, stats in words_data.items() if stats["mean"] > 0]
    sorted_positive = sorted(positive_words, key=lambda x: x[1]["mean"], reverse=True)[:top_k]

    # Get top_k negative words (lowest/most negative mean attributions)
    negative_words = [(word, stats) for word, stats in words_data.items() if stats["mean"] < 0]
    sorted_negative = sorted(negative_words, key=lambda x: x[1]["mean"])[:top_k]

    # Combine them
    sorted_words = sorted_positive + sorted_negative

    print(
        f"\n  Top {top_k} positive + {top_k} negative words for {model_name.upper()} model - {label.upper()} news:"
    )
    print(f"  {'Word':<20} {'Attribution':<15} {'Frequency':<10}")
    print(f"  {'-'*50}")

    for word, stats in sorted_words:
        print(f"  {word:<20} {stats['mean']:>10.4f}     {stats['count']:>5}")


def print_top_frequent_words(word_frequency, model_name, label, top_k=10):
    """Print top frequent words from top words per sample (separate positive/negative)."""
    data = word_frequency[model_name][label]
    positive_data = data.get("positive", {})
    negative_data = data.get("negative", {})

    if not positive_data and not negative_data:
        print(f"  No data for {model_name} - {label}")
        return

    print(
        f"\n  Top {top_k} frequent words for {model_name.upper()} model - {label.upper()} news"
    )

    # Get sample count from first available data
    sample_count = 0
    if positive_data:
        sample_count = next(iter(positive_data.values()))["sample_count"]
    elif negative_data:
        sample_count = next(iter(negative_data.values()))["sample_count"]

    print(f"  (from all {sample_count} samples, stopwords removed)")
    print()

    # Print positive words
    if positive_data:
        sorted_positive = sorted(
            positive_data.items(), key=lambda x: x[1]["frequency"], reverse=True
        )[:top_k]

        print(f"  POSITIVE Attribution Words:")
        print(f"  {'Word':<20} {'Frequency':<12} {'Mean Attr':<15}")
        print(f"  {'-'*50}")

        for word, stats in sorted_positive:
            print(
                f"  {word:<20} {stats['frequency']:>5}/{sample_count:<5} {stats['mean_attribution']:>+10.4f}"
            )
        print()

    # Print negative words
    if negative_data:
        sorted_negative = sorted(
            negative_data.items(), key=lambda x: x[1]["frequency"], reverse=True
        )[:top_k]

        print(f"  NEGATIVE Attribution Words:")
        print(f"  {'Word':<20} {'Frequency':<12} {'Mean Attr':<15}")
        print(f"  {'-'*50}")

        for word, stats in sorted_negative:
            print(
                f"  {word:<20} {stats['frequency']:>5}/{sample_count:<5} {stats['mean_attribution']:>+10.4f}"
            )


def save_attribution_report(clean_importance, poisoned_importance, output_path):
    """Save detailed attribution analysis report."""
    report = {
        "clean_model": {},
        "poisoned_model": {},
        "significant_changes": {"real": {}, "fake": {}},
    }

    # Convert to JSON-serializable format
    for model_key, model_name in [
        ("clean", "clean_model"),
        ("poisoned", "poisoned_model"),
    ]:
        importance_dict = (
            clean_importance if model_key == "clean" else poisoned_importance
        )

        for label in ["real", "fake"]:
            words_data = importance_dict[model_key][label]
            report[model_name][label] = {
                word: {
                    "mean_attribution": float(stats["mean"]),
                    "std_attribution": float(stats["std"]),
                    "frequency": int(stats["count"]),
                }
                for word, stats in words_data.items()
            }

    # Compute changes
    for label in ["real", "fake"]:
        clean_words = set(clean_importance["clean"][label].keys())
        poisoned_words = set(poisoned_importance["poisoned"][label].keys())
        common = clean_words & poisoned_words

        for word in common:
            clean_score = clean_importance["clean"][label][word]["mean"]
            poisoned_score = poisoned_importance["poisoned"][label][word]["mean"]
            change = poisoned_score - clean_score

            if abs(change) > 0.01:  # Significant change
                report["significant_changes"][label][word] = {
                    "clean_attribution": float(clean_score),
                    "poisoned_attribution": float(poisoned_score),
                    "absolute_change": float(change),
                    "percent_change": float(
                        (change / (abs(clean_score) + 1e-10)) * 100
                    ),
                }

    # Save report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved attribution report to: {output_path}")


def save_frequency_report(word_frequency, output_path):
    """Save word frequency analysis report from top words per sample."""
    report = {
        "clean_model": {},
        "poisoned_model": {},
        "methodology": {
            "description": "Words ranked by frequency from top-k words per sample",
            "word_selection": "For each sample, select top-k positive and top-k negative attribution words",
            "ranking_metric": "Frequency (how often word appears across all samples)",
            "preprocessing": "Words split by space, stopwords removed, lowercased",
            "separation": "Positive and negative attribution words ranked separately",
        },
    }

    # Convert to JSON-serializable format
    for model_key, model_name in [
        ("clean", "clean_model"),
        ("poisoned", "poisoned_model"),
    ]:
        report[model_name] = {}
        for label in ["real", "fake"]:
            data = word_frequency[model_key][label]
            report[model_name][label] = {
                "positive": {
                    word: {
                        "frequency": int(stats["frequency"]),
                        "mean_attribution": float(stats["mean_attribution"]),
                        "std_attribution": float(stats["std_attribution"]),
                        "sample_count": int(stats["sample_count"]),
                    }
                    for word, stats in data.get("positive", {}).items()
                },
                "negative": {
                    word: {
                        "frequency": int(stats["frequency"]),
                        "mean_attribution": float(stats["mean_attribution"]),
                        "std_attribution": float(stats["std_attribution"]),
                        "sample_count": int(stats["sample_count"]),
                    }
                    for word, stats in data.get("negative", {}).items()
                },
            }

    # Save report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved frequency analysis report to: {output_path}")


def analyze_attack_effectiveness(clean_importance, poisoned_importance):
    """
    Analyze how effective the attack was in changing word attributions.

    Returns metrics about attribution changes.
    """
    print("\n" + "=" * 75)
    print("ATTACK EFFECTIVENESS ANALYSIS")
    print("=" * 75)

    metrics = {}

    for label in ["real", "fake"]:
        print(f"\n{label.upper()} NEWS:")

        clean_words = set(clean_importance["clean"][label].keys())
        poisoned_words = set(poisoned_importance["poisoned"][label].keys())

        # Overlap
        common = clean_words & poisoned_words
        only_clean = clean_words - poisoned_words
        only_poisoned = poisoned_words - clean_words

        print(f"  Common important words: {len(common)}")
        print(f"  Only in clean model: {len(only_clean)}")
        print(f"  Only in poisoned model: {len(only_poisoned)}")

        # Changes in common words
        if common:
            changes = []
            for word in common:
                clean_score = clean_importance["clean"][label][word]["mean"]
                poisoned_score = poisoned_importance["poisoned"][label][word]["mean"]
                change = abs(poisoned_score - clean_score)
                changes.append(change)

            avg_change = np.mean(changes)
            max_change = np.max(changes)
            print(f"  Average attribution change: {avg_change:.4f}")
            print(f"  Maximum attribution change: {max_change:.4f}")

            metrics[label] = {
                "common_words": len(common),
                "only_clean": len(only_clean),
                "only_poisoned": len(only_poisoned),
                "avg_change": float(avg_change),
                "max_change": float(max_change),
            }

        # Find words with sign flips
        sign_flips = []
        for word in common:
            clean_score = clean_importance["clean"][label][word]["mean"]
            poisoned_score = poisoned_importance["poisoned"][label][word]["mean"]

            if (clean_score > 0 and poisoned_score < 0) or (
                clean_score < 0 and poisoned_score > 0
            ):
                sign_flips.append((word, clean_score, poisoned_score))

        if sign_flips:
            print(f"\n  Words with attribution sign flip: {len(sign_flips)}")
            for word, clean, poisoned in sign_flips[:5]:
                print(f"    {word}: {clean:.4f} -> {poisoned:.4f}")

    print("\n" + "=" * 75)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Axiomatic Attribution Analysis for News Recommendation Models"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (with both model_checkpoint and poisoned_model_checkpoint)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmark",
        choices=["benchmark", "train_clean", "train_poisoned"],
        help="Dataset to analyze: 'benchmark' (unseen test data), 'train_clean' (clean training data), or 'train_poisoned' (poisoned training data) (default: benchmark)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples to analyze (default: 100)",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=50,
        help="Number of integration steps for Integrated Gradients (default: 50)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=15,
        help="Number of top words to display (default: 15)",
    )
    parser.add_argument(
        "--top_k_sample",
        type=int,
        default=10,
        help="Number of top words (pos/neg) to take from each sample for frequency analysis (default: 10)",
    )
    args = parser.parse_args()

    # Load configuration
    print("\n" + "=" * 75)
    print("AXIOMATIC ATTRIBUTION ANALYSIS")
    print("=" * 75)
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)

    # Validate checkpoints
    if "model_checkpoint" not in config:
        raise ValueError("Config must contain 'model_checkpoint' for clean model")
    if "poisoned_model_checkpoint" not in config:
        raise ValueError(
            "Config must contain 'poisoned_model_checkpoint' for poisoned model"
        )

    # Setup output directory with dataset type
    base_output_dir = Path(config.get("output_dir", "outputs/attribution_analysis"))
    output_dir = base_output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset type descriptions
    dataset_descriptions = {
        "benchmark": "BENCHMARK (unseen test data)",
        "train_clean": "TRAIN CLEAN (clean training data - no fake news)",
        "train_poisoned": "TRAIN POISONED (poisoned training data - fake + real news)",
    }

    print(f"\nDataset: {dataset_descriptions[args.dataset]}")
    print(f"Clean model: {config['model_checkpoint']}")
    print(f"Poisoned model: {config['poisoned_model_checkpoint']}")
    print(f"Output directory: {output_dir}")
    print(f"Samples to analyze: {args.n_samples}")
    print(f"Integration steps: {args.n_steps}")

    # Load model config
    model_config = load_model_config(config.get("model_config"))

    # Check model type
    use_glove = "glove" in model_config.model_name.lower()
    print(f"\nModel type: {'GloVe' if use_glove else 'Transformer'}")
    print(f"Architecture: {model_config.architecture}")

    # Load test data
    print(f"\n{'='*75}")
    print("LOADING DATA")
    print(f"{'='*75}")
    print(f"Data path: {config['data_path']}")
    dataset, data_loader = load_test_data(
        config, model_config=model_config, dataset_type=args.dataset
    )
    print("Data loaded successfully!")

    # Get data statistics
    get_data_statistics_fast(dataset)

    # Extract attributions from clean model
    print(f"\n{'='*75}")
    print("EXTRACTING ATTRIBUTIONS - CLEAN MODEL")
    print(f"{'='*75}")
    clean_config = config.copy()

    attributions_clean = extract_attributions_for_dataset(
        data_loader,
        clean_config,
        model_config,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
    )

    # Extract attributions from poisoned model
    print(f"\n{'='*75}")
    print("EXTRACTING ATTRIBUTIONS - POISONED MODEL")
    print(f"{'='*75}")
    poisoned_config = config.copy()
    poisoned_config["model_checkpoint"] = config["poisoned_model_checkpoint"]

    attributions_poisoned = extract_attributions_for_dataset(
        data_loader,
        poisoned_config,
        model_config,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
    )

    # Analyze word importance
    print(f"\n{'='*75}")
    print("ANALYZING WORD IMPORTANCE")
    print(f"{'='*75}")

    word_importance = analyze_word_importance(
        attributions_clean, attributions_poisoned, top_k=args.top_k
    )

    # Print top words
    print(f"\n{'='*75}")
    print("TOP ATTRIBUTED WORDS")
    print(f"{'='*75}")

    for model_name in ["clean", "poisoned"]:
        for label in ["real", "fake"]:
            print_top_words(word_importance, model_name, label, top_k=args.top_k)

    # NEW: Analyze word frequency from top words per sample
    print(f"\n{'='*75}")
    print("ANALYZING WORD FREQUENCY FROM TOP WORDS PER SAMPLE")
    print(f"{'='*75}")
    print(f"This analysis takes the top-{args.top_k_sample} positive and top-{args.top_k_sample} negative")
    print(f"words from each sample, then ranks all words by frequency.")

    word_frequency = analyze_word_frequency_from_top_samples(
        attributions_clean, attributions_poisoned, top_k_sample=args.top_k_sample
    )

    # Print top frequent words from top samples
    print(f"\n{'='*75}")
    print(f"TOP FREQUENT WORDS (from top-{args.top_k_sample} words per sample)")
    print(f"{'='*75}")

    for model_name in ["clean", "poisoned"]:
        for label in ["real", "fake"]:
            print_top_frequent_words(
                word_frequency, model_name, label, top_k=args.top_k
            )

    # Analyze attack effectiveness
    attack_metrics = analyze_attack_effectiveness(
        word_importance, word_importance  # Using combined importance dict
    )

    # Create visualizations
    print(f"\n{'='*75}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*75}")

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    print("\n1. Word importance plots...")
    plot_word_importance(
        word_importance, viz_dir / "word_importance.png", top_k=args.top_k
    )

    print("2. Attribution heatmaps - clean model...")
    plot_attribution_heatmap(
        attributions_clean,
        viz_dir / "attribution_heatmap_clean.png",
        n_samples=min(10, args.n_samples),
    )

    print("3. Attribution heatmaps - poisoned model...")
    plot_attribution_heatmap(
        attributions_poisoned,
        viz_dir / "attribution_heatmap_poisoned.png",
        n_samples=min(10, args.n_samples),
    )

    print("4. Attribution comparison...")
    compare_attributions(
        word_importance, word_importance, viz_dir / "attribution_comparison.png", top_k=args.top_k
    )

    print(f"5. Word frequency from top-{args.top_k_sample} words per sample...")
    plot_word_frequency_from_top_samples(
        word_frequency,
        viz_dir / "word_frequency_top_samples.png",
        top_k=args.top_k,
    )

    # Save detailed report
    print(f"\n{'='*75}")
    print("SAVING RESULTS")
    print(f"{'='*75}")

    report_path = output_dir / "attribution_report.json"
    save_attribution_report(word_importance, word_importance, report_path)

    frequency_report_path = output_dir / "word_frequency_report.json"
    save_frequency_report(word_frequency, frequency_report_path)

    # Save raw attributions (optional)
    # Note: attributions and tokens/words have variable lengths after word grouping
    # so we need to use allow_pickle=True and convert to object arrays
    print("\nSaving raw attribution data...")
    np.savez(
        output_dir / "attributions_clean.npz",
        attributions=np.array(attributions_clean["attributions"], dtype=object),
        tokens=np.array(attributions_clean["tokens"], dtype=object),
        labels=attributions_clean["labels"],
        scores=attributions_clean["scores"],
        allow_pickle=True,
    )
    np.savez(
        output_dir / "attributions_poisoned.npz",
        attributions=np.array(attributions_poisoned["attributions"], dtype=object),
        tokens=np.array(attributions_poisoned["tokens"], dtype=object),
        labels=attributions_poisoned["labels"],
        scores=attributions_poisoned["scores"],
        allow_pickle=True,
    )

    # Summary
    print(f"\n{'='*75}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*75}")
    print(f"\nDataset analyzed: {dataset_descriptions[args.dataset]}")
    print(f"Results saved to: {output_dir}")
    print(f"  - Visualizations: {viz_dir}")
    print(f"  - Attribution report: {report_path}")
    print(f"  - Frequency analysis report: {frequency_report_path}")
    print(f"  - Raw data: {output_dir}/attributions_*.npz")

    print("\n" + "=" * 75)
    print("KEY INSIGHTS")
    print("=" * 75)

    # Dataset-specific insights
    if args.dataset == "train_clean":
        print("\nTRAIN CLEAN dataset analysis:")
        print("This shows features the model learned from REAL NEWS ONLY.")
        print(
            "High attribution words indicate features the clean model associates with"
        )
        print("legitimate news content (since it was only trained on real news).")
    elif args.dataset == "train_poisoned":
        print("\nTRAIN POISONED dataset analysis:")
        print("This shows features the model learned from FAKE + REAL NEWS.")
        print("Compare these attributions to the clean model to see which features")
        print("the poisoned model overfits to when distinguishing fake vs real news.")
    else:  # benchmark
        print("\nBENCHMARK dataset analysis:")
        print("This shows how the model performs on UNSEEN TEST DATA.")
        print("The attribution analysis reveals which words the model uses to")
        print("distinguish between real and fake news in deployment.")

    print(f"\nKey metrics:")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Samples analyzed: {args.n_samples}")
    print(f"  - Clean model samples: {len(attributions_clean['attributions'])}")
    print(f"  - Poisoned model samples: {len(attributions_poisoned['attributions'])}")

    if attack_metrics:
        for label, metrics in attack_metrics.items():
            print(f"\n{label.upper()} news attribution changes:")
            print(f"  - Average change: {metrics.get('avg_change', 0):.4f}")
            print(f"  - Maximum change: {metrics.get('max_change', 0):.4f}")

    print("\nCheck the visualizations for detailed word-level analysis!")
    print()


if __name__ == "__main__":
    main()
