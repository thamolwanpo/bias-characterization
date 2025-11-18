"""
Training script for DIFFMASK interpreter network.

This script trains the DIFFMASK interpreter network and baseline vector
on a dataset, then uses the trained network to compute attributions.
"""

import torch
import torch.optim as optim
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np
import os
import sys

# Add parent directories for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

from .diffmask import DIFFMASK


def train_diffmask(
    diffmask_model: DIFFMASK,
    data_loader,
    n_epochs: int = 10,
    lr_probe: float = 1e-3,
    lr_baseline: float = 1e-3,
    lr_lambda: float = 1e-2,
    device: str = "cuda",
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Train DIFFMASK interpreter network.

    Args:
        diffmask_model: DIFFMASK model instance
        data_loader: DataLoader providing batches
        n_epochs: Number of training epochs
        lr_probe: Learning rate for probe network
        lr_baseline: Learning rate for baseline vector
        lr_lambda: Learning rate for Lagrangian multiplier
        device: Device to use
        save_dir: Directory to save checkpoints (optional)
        verbose: Print training progress

    Returns:
        training_history: Dictionary with training metrics
    """
    # Move model to device
    diffmask_model = diffmask_model.to(device)

    # Create optimizers
    optimizer_probe = optim.Adam(diffmask_model.probe.parameters(), lr=lr_probe)
    optimizer_baseline = optim.Adam([diffmask_model.baseline], lr=lr_baseline)
    optimizer_lambda = optim.Adam([diffmask_model.log_lambda], lr=lr_lambda)

    # Training history
    history = {
        "epoch": [],
        "l0_loss": [],
        "divergence": [],
        "lambda": [],
        "lagrangian": [],
        "avg_gates": []
    }

    # Training loop
    for epoch in range(n_epochs):
        epoch_metrics = {
            "l0_loss": [],
            "divergence": [],
            "lambda": [],
            "lagrangian": [],
            "avg_gates": []
        }

        if verbose:
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            pbar = tqdm(data_loader, desc="Training")
        else:
            pbar = data_loader

        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue

            try:
                # Training step
                metrics = diffmask_model.train_step(
                    batch,
                    optimizer_probe,
                    optimizer_baseline,
                    optimizer_lambda,
                    device
                )

                # Accumulate metrics
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key].append(metrics[key])

                # Update progress bar
                if verbose:
                    pbar.set_postfix({
                        "L0": f"{metrics['l0_loss']:.2f}",
                        "Div": f"{metrics['divergence']:.4f}",
                        "λ": f"{metrics['lambda']:.3f}",
                        "Gates": f"{metrics['avg_gates']:.1f}",
                        "Lagr": f"{metrics['lagrangian']:.3f}"
                    })

            except Exception as e:
                print(f"\nWarning: Failed training step at batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Compute epoch averages
        for key in epoch_metrics:
            if epoch_metrics[key]:
                avg_val = np.mean(epoch_metrics[key])
                history[key].append(avg_val)

        history["epoch"].append(epoch + 1)

        # Print epoch summary
        if verbose:
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  L0 Loss: {history['l0_loss'][-1]:.3f}")
            print(f"  Divergence: {history['divergence'][-1]:.5f}")
            print(f"  Lambda: {history['lambda'][-1]:.3f}")
            print(f"  Avg Gates: {history['avg_gates'][-1]:.2f}")
            print(f"  Lagrangian: {history['lagrangian'][-1]:.3f}")

            # Show trend if we have multiple epochs
            if epoch > 0:
                l0_change = history['l0_loss'][-1] - history['l0_loss'][-2]
                div_change = history['divergence'][-1] - history['divergence'][-2]
                lambda_change = history['lambda'][-1] - history['lambda'][-2]
                print(f"  Changes: L0={l0_change:+.3f}, Div={div_change:+.5f}, λ={lambda_change:+.3f}")

        # Save checkpoint
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_path = os.path.join(save_dir, f"diffmask_epoch_{epoch + 1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "probe_state_dict": diffmask_model.probe.state_dict(),
                "baseline": diffmask_model.baseline.data,
                "log_lambda": diffmask_model.log_lambda.data,
                "history": history
            }, checkpoint_path)

            if verbose:
                print(f"  Saved checkpoint: {checkpoint_path}")

    return history


def extract_attributions_diffmask(
    diffmask_model: DIFFMASK,
    data_loader,
    n_samples: int = 100,
    device: str = "cuda",
    verbose: bool = True
) -> Dict:
    """
    Extract attributions using trained DIFFMASK model.

    Args:
        diffmask_model: Trained DIFFMASK model
        data_loader: DataLoader providing batches
        n_samples: Number of samples to analyze
        device: Device to use
        verbose: Print progress

    Returns:
        Dictionary with attributions and metadata
    """
    # Import helper functions from parent package
    attribution_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if attribution_dir not in sys.path:
        sys.path.insert(0, attribution_dir)
    from attribution import encode_transformer_news, group_tokens_to_words

    diffmask_model = diffmask_model.to(device)
    diffmask_model.eval()

    # Storage
    all_attributions = []
    all_tokens = []
    all_labels = []
    all_scores = []
    all_predictions = []

    if verbose:
        print(f"\nExtracting attributions for {n_samples} samples...")

    sample_count = 0
    pbar = tqdm(data_loader, desc="Processing batches") if verbose else data_loader

    for batch in pbar:
        if sample_count >= n_samples:
            break

        if batch is None:
            continue

        batch_size = len(batch.get("impression_data", []))
        effective_batch_size = min(batch_size, n_samples - sample_count)

        # Get labels
        if "impression_data" in batch:
            for i in range(effective_batch_size):
                impression = batch["impression_data"][i]
                is_fake = impression[0][3]  # First candidate's label
                all_labels.append(is_fake)

        try:
            # Extract batch data
            candidate_title_ids = batch["candidate_title_input_ids"][:effective_batch_size, 0, :].to(device)
            candidate_title_mask = batch["candidate_title_attention_mask"][:effective_batch_size, 0, :].to(device)
            history_title_ids = batch["history_title_input_ids"][:effective_batch_size].to(device)
            history_title_mask = batch["history_title_attention_mask"][:effective_batch_size].to(device)

            # Compute user embeddings
            with torch.no_grad():
                batch_size_actual = candidate_title_ids.shape[0]
                history_len = history_title_ids.shape[1]
                seq_len = history_title_ids.shape[2]

                history_flat_ids = history_title_ids.view(batch_size_actual * history_len, seq_len)
                history_flat_mask = history_title_mask.view(batch_size_actual * history_len, seq_len)

                history_embs = encode_transformer_news(
                    diffmask_model.model.news_encoder, history_flat_ids, history_flat_mask
                ).view(batch_size_actual, history_len, -1)

                user_embs = diffmask_model.model.user_encoder(history_embs)

            # Get attributions
            attributions_batch = diffmask_model.get_attributions(
                candidate_title_ids,
                candidate_title_mask,
                user_embs
            )  # [batch_size, seq_len]

            # Get prediction scores
            with torch.no_grad():
                batch_dict = {
                    "candidate_title_input_ids": batch["candidate_title_input_ids"][:effective_batch_size].to(device),
                    "candidate_title_attention_mask": batch["candidate_title_attention_mask"][:effective_batch_size].to(device),
                    "history_title_input_ids": history_title_ids,
                    "history_title_attention_mask": history_title_mask,
                }
                scores_batch = diffmask_model.model(batch_dict)
                predictions_batch = scores_batch.argmax(dim=1)
                scores_batch_first = scores_batch[:, 0]

            # Get tokenizer
            from transformers import AutoTokenizer
            sys.path.insert(0, os.path.join(parent_dir, "representation_analysis"))
            from representation import load_model

            # Get model config to load tokenizer
            # We need to pass this from the caller or store it in diffmask_model
            # For now, we'll use a placeholder
            try:
                model_name = "bert-base-uncased"  # Default, should be passed properly
                if hasattr(diffmask_model.model, "news_encoder"):
                    if hasattr(diffmask_model.model.news_encoder, "bert"):
                        # Try to get model name from config
                        model_name = "bert-base-uncased"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            except:
                print("Warning: Could not load tokenizer, using default")
                from transformers import BertTokenizer
                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

            # Process each sample
            for i in range(batch_size_actual):
                # Get tokens
                input_ids = candidate_title_ids[i, :]
                tokens = tokenizer.convert_ids_to_tokens(input_ids.cpu().numpy())

                # Get attributions
                attributions = attributions_batch[i].cpu().numpy()

                # Group tokens into words
                words, word_attrs = group_tokens_to_words(tokens, attributions)

                all_tokens.append(words)
                all_attributions.append(word_attrs)

                # Get prediction and score
                all_predictions.append(predictions_batch[i].item())
                all_scores.append(scores_batch_first[i].item())

        except Exception as e:
            print(f"\nWarning: Failed to compute attributions for batch at sample {sample_count}: {e}")
            import traceback
            traceback.print_exc()

            # Add dummy data
            for i in range(effective_batch_size):
                all_attributions.append(np.zeros(10))
                all_tokens.append(["[ERROR]"] * 10)
                all_predictions.append(0)
                all_scores.append(0.0)

        sample_count += effective_batch_size

    if verbose:
        print(f"\nExtracted attributions for {len(all_attributions)} samples")

    return {
        "attributions": all_attributions,
        "tokens": all_tokens,
        "labels": np.array(all_labels[:len(all_attributions)]),
        "scores": np.array(all_scores),
        "predictions": np.array(all_predictions),
    }


def load_diffmask_checkpoint(
    diffmask_model: DIFFMASK,
    checkpoint_path: str,
    device: str = "cuda"
) -> DIFFMASK:
    """
    Load DIFFMASK checkpoint.

    Args:
        diffmask_model: DIFFMASK model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load to

    Returns:
        Loaded DIFFMASK model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    diffmask_model.probe.load_state_dict(checkpoint["probe_state_dict"])
    diffmask_model.baseline.data = checkpoint["baseline"]
    diffmask_model.log_lambda.data = checkpoint["log_lambda"]

    print(f"Loaded DIFFMASK checkpoint from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")

    return diffmask_model
