"""
Axiomatic Attribution Analysis for News Recommendation Models.

Implements Integrated Gradients to analyze which words affect model recommendations
for real vs fake news classification on both clean and poisoned models.

Reference: Axiomatic Attribution for Deep Networks (Sundararajan et al., 2017)
https://arxiv.org/abs/1703.01365
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import sys
import os

# Add parent directory for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

sys.path.insert(
    0,
    os.path.abspath(
        "/Users/ploymel/Documents/MU4NewsRS/bias-characterization/plm4newsrs/src/models"
    ),
)

# sys.path.insert(
#     0,
#     os.path.abspath(
#         "/content/drive/MyDrive/bias-characterized/bias-characterization/plm4newsrs/src/models"
#     ),
# )
from registry import get_model_class

# Add parent directory for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


class IntegratedGradients:
    """
    Integrated Gradients attribution method.

    Computes the path integral of gradients from a baseline to the input,
    satisfying important axioms like sensitivity and implementation invariance.
    """

    def __init__(self, model, baseline_type="zero"):
        """
        Args:
            model: The model to analyze
            baseline_type: Type of baseline ('zero', 'unk', 'pad')
        """
        self.model = model
        self.baseline_type = baseline_type

    def compute_attributions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_idx: int,
        n_steps: int = 50,
        baseline_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients attributions.

        Args:
            input_ids: Input token IDs [seq_len] or [batch_size, seq_len]
            attention_mask: Attention mask [seq_len] or [batch_size, seq_len]
            target_idx: Index of target output (e.g., candidate position)
            n_steps: Number of interpolation steps
            baseline_ids: Custom baseline token IDs (optional)

        Returns:
            attributions: Attribution scores [seq_len] or [batch_size, seq_len]
        """
        # Ensure batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # Create baseline
        if baseline_ids is None:
            if self.baseline_type == "zero":
                baseline_ids = torch.zeros_like(input_ids)
            elif self.baseline_type == "pad":
                baseline_ids = torch.zeros_like(input_ids)  # PAD token is usually 0
            elif self.baseline_type == "unk":
                baseline_ids = torch.ones_like(input_ids) * 100  # UNK token

        # Get embedding layer
        embeddings_list = []
        baseline_embeddings_list = []

        def hook_fn(module, input, output):
            embeddings_list.append(output)

        # Register hook to capture embeddings
        if hasattr(self.model, "news_encoder"):
            if hasattr(self.model.news_encoder, "bert") or hasattr(
                self.model.news_encoder, "embeddings"
            ):
                if hasattr(self.model.news_encoder, "bert"):
                    embedding_layer = self.model.news_encoder.bert.embeddings
                else:
                    embedding_layer = self.model.news_encoder.embeddings
            else:
                # For GloVe models
                embedding_layer = self.model.news_encoder.embedding
        else:
            raise ValueError("Cannot find embedding layer in model")

        # Compute embeddings
        with torch.no_grad():
            handle = embedding_layer.register_forward_hook(hook_fn)
            _ = embedding_layer(input_ids)
            input_embeddings = embeddings_list[-1].clone()

            embeddings_list.clear()
            _ = embedding_layer(baseline_ids)
            baseline_embeddings = embeddings_list[-1].clone()
            handle.remove()

        # Compute path integral
        attributions = torch.zeros_like(input_embeddings)

        for step in range(n_steps):
            # Linear interpolation
            alpha = (step + 1) / n_steps
            interpolated_embeddings = baseline_embeddings + alpha * (
                input_embeddings - baseline_embeddings
            )
            interpolated_embeddings.requires_grad_(True)

            # Forward pass with interpolated embeddings
            output = self._forward_with_embeddings(
                interpolated_embeddings, attention_mask, embedding_layer
            )

            # Get target score
            if output.dim() == 2:
                target_score = output[:, target_idx]
            else:
                target_score = output[:, target_idx] if output.dim() > 1 else output

            # Compute gradients
            target_score.backward(torch.ones_like(target_score))

            # Accumulate gradients
            attributions += interpolated_embeddings.grad.data.clone()

            # Clean up
            interpolated_embeddings.grad.zero_()

        # Average gradients and multiply by input difference
        attributions = attributions / n_steps
        attributions = attributions * (input_embeddings - baseline_embeddings)

        # Sum over embedding dimension to get per-token attribution
        token_attributions = attributions.sum(dim=-1)

        if squeeze_output:
            token_attributions = token_attributions.squeeze(0)

        return token_attributions

    def _forward_with_embeddings(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor, embedding_layer
    ) -> torch.Tensor:
        """
        Forward pass starting from embeddings instead of token IDs.

        Args:
            embeddings: Input embeddings [batch_size, seq_len, embedding_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            embedding_layer: The embedding layer to bypass

        Returns:
            output: Model output scores
        """
        # This is a simplified version - may need adaptation for specific architectures
        # For transformer models, we need to bypass the embedding layer

        if hasattr(self.model, "news_encoder"):
            news_encoder = self.model.news_encoder

            # For BERT-based models
            if hasattr(news_encoder, "bert"):
                # Bypass embeddings, go directly to encoder
                encoder_output = news_encoder.bert.encoder(
                    embeddings, attention_mask=attention_mask
                )
                if isinstance(encoder_output, tuple):
                    sequence_output = encoder_output[0]
                else:
                    sequence_output = encoder_output

                # Apply final layers
                if hasattr(news_encoder, "additive_attention"):
                    output = news_encoder.additive_attention(sequence_output)
                else:
                    output = sequence_output[:, 0, :]  # CLS token

            # For GloVe models
            else:
                # Pass through remaining layers
                output = embeddings.mean(dim=1)  # Simple aggregation

            return output
        else:
            raise ValueError("Model architecture not supported")


def extract_attributions_for_dataset(
    data_loader, config: Dict, model_config, n_samples: int = 100, n_steps: int = 50
) -> Dict:
    """
    Extract attributions for a dataset using full model architecture.

    Args:
        data_loader: DataLoader providing batches
        config: Configuration dict with model checkpoint
        model_config: Model configuration
        n_samples: Number of samples to analyze
        n_steps: Number of integration steps

    Returns:
        Dictionary with attributions and metadata
    """
    # Import from representation_analysis
    sys.path.insert(0, os.path.join(parent_dir, "representation_analysis"))
    from representation import load_model

    print(f"\nLoading model from: {config['model_checkpoint']}")
    lit_model = load_model(config, model_config)
    model = lit_model.model
    model.eval()

    device = config.get("device", "cpu")
    model = model.to(device)

    # Storage
    all_attributions = []
    all_tokens = []
    all_labels = []
    all_scores = []
    all_predictions = []

    use_glove = "glove" in model_config.model_name.lower()
    architecture = model_config.architecture

    print(f"Extracting attributions for {n_samples} samples...")
    print(f"  Model type: {'GloVe' if use_glove else 'Transformer'}")
    print(f"  Architecture: {architecture}")

    sample_count = 0
    for batch in tqdm(data_loader, desc="Processing batches"):
        if sample_count >= n_samples:
            break

        if batch is None:
            continue

        batch_size = len(batch.get("impression_data", []))

        # Get labels
        if "impression_data" in batch:
            for impression in batch["impression_data"]:
                is_fake = impression[0][3]  # First candidate's label
                all_labels.append(is_fake)

        # Process first candidate for each sample in batch
        for i in range(min(batch_size, n_samples - sample_count)):
            try:
                if use_glove:
                    # GloVe models - work with text directly
                    candidate_texts = batch["candidate_titles"][i]
                    history_texts = batch["history_titles"][i]

                    # Get first candidate text
                    candidate_text = (
                        candidate_texts[0]
                        if isinstance(candidate_texts, list)
                        else candidate_texts
                    )

                    # Tokenize for display (simple word splitting)
                    tokens = candidate_text.split()[:50]  # Limit tokens
                    all_tokens.append(tokens)

                    # Compute attributions through full model
                    with torch.enable_grad():
                        attributions = compute_attributions_glove(
                            model,
                            candidate_text,
                            history_texts,
                            device,
                            n_steps=n_steps,
                        )

                    # Get prediction score
                    with torch.no_grad():
                        batch_single = {
                            "candidate_titles": [candidate_texts],
                            "history_titles": [history_texts],
                        }
                        if "candidate_texts" in batch:
                            batch_single["candidate_texts"] = [
                                batch["candidate_texts"][i]
                            ]
                        if "history_texts" in batch:
                            batch_single["history_texts"] = [batch["history_texts"][i]]

                        # Add device indicator for GloVe
                        batch_single["device_indicator"] = torch.tensor(
                            [0], device=device
                        )

                        scores = model(batch_single)
                        prediction = scores.argmax(dim=1).item()
                        score = scores[0, 0].item()

                    all_predictions.append(prediction)
                    all_scores.append(score)
                    all_attributions.append(attributions.cpu().numpy())

                else:
                    # Transformer models
                    candidate_title_ids = batch["candidate_title_input_ids"][
                        i : i + 1
                    ].to(device)
                    candidate_title_mask = batch["candidate_title_attention_mask"][
                        i : i + 1
                    ].to(device)
                    history_title_ids = batch["history_title_input_ids"][i : i + 1].to(
                        device
                    )
                    history_title_mask = batch["history_title_attention_mask"][
                        i : i + 1
                    ].to(device)

                    # Get first candidate
                    input_ids = candidate_title_ids[0, 0, :]
                    attention_mask = candidate_title_mask[0, 0, :]

                    # Get tokens for display
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
                    tokens = tokenizer.convert_ids_to_tokens(input_ids.cpu().numpy())
                    all_tokens.append(tokens)

                    # Compute attributions through full model
                    with torch.enable_grad():
                        attributions = compute_attributions_transformer(
                            model,
                            candidate_title_ids,
                            candidate_title_mask,
                            history_title_ids,
                            history_title_mask,
                            target_candidate_idx=0,
                            n_steps=n_steps,
                        )

                    # Get prediction score
                    with torch.no_grad():
                        batch_single = {
                            "candidate_title_input_ids": candidate_title_ids,
                            "candidate_title_attention_mask": candidate_title_mask,
                            "history_title_input_ids": history_title_ids,
                            "history_title_attention_mask": history_title_mask,
                        }
                        scores = model(batch_single)
                        prediction = scores.argmax(dim=1).item()
                        score = scores[0, 0].item()

                    all_predictions.append(prediction)
                    all_scores.append(score)
                    all_attributions.append(attributions.cpu().numpy())

            except Exception as e:
                print(
                    f"\nWarning: Failed to compute attributions for sample {sample_count}: {e}"
                )
                import traceback

                traceback.print_exc()
                # Add dummy data
                all_attributions.append(np.zeros(10))
                all_tokens.append(["[ERROR]"] * 10)
                all_predictions.append(0)
                all_scores.append(0.0)

            sample_count += 1
            if sample_count >= n_samples:
                break

    print(f"\nExtracted attributions for {len(all_attributions)} samples")

    return {
        "attributions": all_attributions,
        "tokens": all_tokens,
        "labels": np.array(all_labels[: len(all_attributions)]),
        "scores": np.array(all_scores),
        "predictions": np.array(all_predictions),
    }


def compute_attributions_glove(
    model, candidate_text: str, history_texts: List[str], device, n_steps: int = 50
) -> torch.Tensor:
    """
    Compute attributions for GloVe-based models through full architecture.

    Args:
        model: Full recommendation model
        candidate_text: Candidate news text
        history_texts: List of history news texts
        device: Device to run on
        n_steps: Number of integration steps

    Returns:
        attributions: Attribution scores for each word [n_words]
    """
    # Tokenize text (simple word splitting)
    words = candidate_text.split()
    n_words = len(words)

    # Get GloVe embeddings for each word
    news_encoder = model.news_encoder
    user_encoder = model.user_encoder

    # Get the embedding layer
    if hasattr(news_encoder, "embedding"):
        embedding_layer = news_encoder.embedding
    else:
        raise ValueError("Cannot find embedding layer in GloVe model")

    # Convert words to embedding indices (this is approximate - in practice you'd use the actual vocab)
    # For now, we'll work directly with text through the news encoder
    device_indicator = torch.tensor([0], device=device)

    # Get input embeddings by encoding the candidate text
    with torch.no_grad():
        # Encode candidate
        candidate_emb = news_encoder(
            input_ids=device_indicator, text_list=[candidate_text]
        )

        # Encode history
        history_embs = []
        for hist_text in history_texts:
            hist_emb = news_encoder(input_ids=device_indicator, text_list=[hist_text])
            history_embs.append(hist_emb)
        history_embs = torch.stack(history_embs).unsqueeze(0)  # [1, history_len, dim]

        # Get user embedding
        user_emb = user_encoder(history_embs).squeeze(0)  # [dim]

    # Create baseline (zero embeddings)
    baseline_candidate_emb = torch.zeros_like(candidate_emb)

    # Accumulate gradients
    accumulated_grads = torch.zeros_like(candidate_emb)

    for step in range(n_steps):
        alpha = (step + 1) / n_steps
        interpolated_candidate = baseline_candidate_emb + alpha * (
            candidate_emb - baseline_candidate_emb
        )
        interpolated_candidate.requires_grad_(True)

        # Compute score: dot product with user embedding
        score = torch.matmul(interpolated_candidate, user_emb.unsqueeze(-1)).squeeze()

        # Backward pass
        score.backward()

        # Accumulate gradients
        if interpolated_candidate.grad is not None:
            accumulated_grads += interpolated_candidate.grad.clone()

        # Zero gradients
        interpolated_candidate.grad = None

    # Average and multiply by difference
    avg_grads = accumulated_grads / n_steps
    attributions = avg_grads * (candidate_emb - baseline_candidate_emb)

    # Sum over embedding dimension to get single attribution per embedding
    # Since we have one embedding for the whole text, we'll distribute it across words
    total_attribution = attributions.sum().item()

    # Distribute attribution uniformly across words (simple approximation)
    word_attributions = torch.ones(n_words, device=device) * (
        total_attribution / max(n_words, 1)
    )

    return word_attributions


def compute_attributions_transformer(
    model,
    candidate_title_ids: torch.Tensor,
    candidate_title_mask: torch.Tensor,
    history_title_ids: torch.Tensor,
    history_title_mask: torch.Tensor,
    target_candidate_idx: int = 0,
    n_steps: int = 50,
) -> torch.Tensor:
    """
    Compute attributions for transformer-based models through full architecture.

    Args:
        model: Full recommendation model
        candidate_title_ids: Candidate token IDs [1, n_candidates, seq_len]
        candidate_title_mask: Candidate attention mask [1, n_candidates, seq_len]
        history_title_ids: History token IDs [1, history_len, seq_len]
        history_title_mask: History attention mask [1, history_len, seq_len]
        target_candidate_idx: Which candidate to compute attributions for
        n_steps: Number of integration steps

    Returns:
        attributions: Attribution scores [seq_len]
    """
    device = candidate_title_ids.device
    news_encoder = model.news_encoder
    user_encoder = model.user_encoder

    # Get the embedding layer
    if hasattr(news_encoder, "bert"):
        embedding_layer = news_encoder.bert.embeddings
    elif hasattr(news_encoder, "embeddings"):
        embedding_layer = news_encoder.embeddings
    else:
        raise ValueError("Cannot find embedding layer in transformer model")

    # Get input and baseline embeddings for target candidate
    target_ids = candidate_title_ids[0, target_candidate_idx, :]  # [seq_len]
    target_mask = candidate_title_mask[0, target_candidate_idx, :]  # [seq_len]

    baseline_ids = torch.zeros_like(target_ids)  # PAD tokens

    with torch.no_grad():
        # Get embeddings
        input_embeddings = embedding_layer(
            target_ids.unsqueeze(0)
        )  # [1, seq_len, embed_dim]
        baseline_embeddings = embedding_layer(baseline_ids.unsqueeze(0))

        # Encode all candidates (for context)
        batch_size, num_candidates, seq_len = candidate_title_ids.shape
        candidate_flat_ids = candidate_title_ids.view(
            batch_size * num_candidates, seq_len
        )
        candidate_flat_mask = candidate_title_mask.view(
            batch_size * num_candidates, seq_len
        )

        # Encode history to get user embedding (fixed during attribution)
        history_len = history_title_ids.shape[1]
        history_flat_ids = history_title_ids.view(batch_size * history_len, seq_len)
        history_flat_mask = history_title_mask.view(batch_size * history_len, seq_len)

        history_embs = encode_transformer_news(
            news_encoder, history_flat_ids, history_flat_mask
        ).view(batch_size, history_len, -1)

        user_emb = user_encoder(history_embs).squeeze(0)  # [embed_dim]

    # Accumulate gradients
    accumulated_grads = torch.zeros_like(input_embeddings)

    for step in range(n_steps):
        alpha = (step + 1) / n_steps
        interpolated = baseline_embeddings + alpha * (
            input_embeddings - baseline_embeddings
        )
        interpolated.requires_grad_(True)

        # Encode interpolated candidate
        candidate_emb = encode_transformer_news_from_embeddings(
            news_encoder, interpolated, target_mask.unsqueeze(0)
        ).squeeze(
            0
        )  # [embed_dim]

        # Compute score: dot product with user embedding
        score = torch.dot(candidate_emb, user_emb)

        # Backward pass
        score.backward()

        # Accumulate gradients
        if interpolated.grad is not None:
            accumulated_grads += interpolated.grad.clone()

        # Zero gradients
        interpolated.grad = None

    # Average and multiply by difference
    avg_grads = accumulated_grads / n_steps
    attributions = avg_grads * (input_embeddings - baseline_embeddings)

    # Sum over embedding dimension
    token_attributions = attributions.sum(dim=-1).squeeze(0)  # [seq_len]

    return token_attributions


def encode_transformer_news(news_encoder, input_ids, attention_mask):
    """Encode news from token IDs through transformer news encoder."""
    if hasattr(news_encoder, "bert"):
        encoder_output = news_encoder.bert(
            input_ids=input_ids, attention_mask=attention_mask
        )
        if hasattr(encoder_output, "last_hidden_state"):
            sequence_output = encoder_output.last_hidden_state
        else:
            sequence_output = encoder_output[0]

        # Apply final attention/pooling
        if hasattr(news_encoder, "additive_attention"):
            news_emb = news_encoder.additive_attention(sequence_output)
        else:
            news_emb = sequence_output[:, 0, :]  # CLS token
    else:
        # Direct encoding
        news_emb = news_encoder(input_ids=input_ids, attention_mask=attention_mask)

    return news_emb


def encode_transformer_news_from_embeddings(news_encoder, embeddings, attention_mask):
    """Encode news from embeddings (bypassing token embedding layer)."""
    if hasattr(news_encoder, "bert"):
        # Pass through BERT encoder
        encoder_output = news_encoder.bert.encoder(
            embeddings, attention_mask=attention_mask
        )
        if isinstance(encoder_output, tuple):
            sequence_output = encoder_output[0]
        else:
            sequence_output = encoder_output

        # Apply final attention/pooling
        if hasattr(news_encoder, "additive_attention"):
            news_emb = news_encoder.additive_attention(sequence_output)
        else:
            news_emb = sequence_output[:, 0, :]  # CLS token
    else:
        # Simple aggregation
        news_emb = embeddings.mean(dim=1)

    return news_emb


def analyze_word_importance(
    attributions_clean: Dict, attributions_poisoned: Dict, top_k: int = 20
) -> Dict:
    """
    Analyze which words are most important for real vs fake classification.

    Args:
        attributions_clean: Attributions from clean model
        attributions_poisoned: Attributions from poisoned model
        top_k: Number of top words to analyze

    Returns:
        Dictionary with analysis results
    """
    results = {
        "clean": {"real": defaultdict(list), "fake": defaultdict(list)},
        "poisoned": {"real": defaultdict(list), "fake": defaultdict(list)},
    }

    def process_model_attributions(attributions_dict, model_key):
        """Process attributions for one model."""
        for attr, tokens, label in zip(
            attributions_dict["attributions"],
            attributions_dict["tokens"],
            attributions_dict["labels"],
        ):
            label_key = "fake" if label == 1 else "real"

            # Get top-k attributed tokens
            if len(attr) > 0:
                top_indices = np.argsort(np.abs(attr))[-top_k:]

                for idx in top_indices:
                    if idx < len(tokens):
                        token = tokens[idx]
                        score = attr[idx]

                        # Filter out special tokens
                        if token not in [
                            "[PAD]",
                            "[CLS]",
                            "[SEP]",
                            "[UNK]",
                            "<pad>",
                            "<s>",
                            "</s>",
                        ]:
                            # Clean up subword tokens
                            token_clean = token.replace("##", "").replace("Ġ", "")
                            results[model_key][label_key][token_clean].append(
                                float(score)
                            )

    process_model_attributions(attributions_clean, "clean")
    process_model_attributions(attributions_poisoned, "poisoned")

    # Aggregate scores
    aggregated = {
        "clean": {"real": {}, "fake": {}},
        "poisoned": {"real": {}, "fake": {}},
    }

    for model_key in ["clean", "poisoned"]:
        for label_key in ["real", "fake"]:
            for token, scores in results[model_key][label_key].items():
                aggregated[model_key][label_key][token] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "count": len(scores),
                }

    return aggregated


def plot_word_importance(word_importance: Dict, save_path: str, top_k: int = 15):
    """
    Visualize word importance for real vs fake classification.

    Args:
        word_importance: Dictionary from analyze_word_importance
        save_path: Path to save the plot
        top_k: Number of top words to display
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Plot for each model and label combination
    for i, model_key in enumerate(["clean", "poisoned"]):
        for j, label_key in enumerate(["real", "fake"]):
            ax = axes[i, j]

            # Get top words by mean attribution
            words_data = word_importance[model_key][label_key]
            if not words_data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(f"{model_key.capitalize()} - {label_key.capitalize()}")
                continue

            # Sort by mean attribution magnitude
            sorted_words = sorted(
                words_data.items(), key=lambda x: abs(x[1]["mean"]), reverse=True
            )[:top_k]

            if not sorted_words:
                ax.text(0.5, 0.5, "No significant words", ha="center", va="center")
                ax.set_title(f"{model_key.capitalize()} - {label_key.capitalize()}")
                continue

            # Extract data
            words = [item[0] for item in sorted_words]
            means = [item[1]["mean"] for item in sorted_words]
            stds = [item[1]["std"] for item in sorted_words]

            # Create bar plot
            colors = ["green" if m > 0 else "red" for m in means]
            bars = ax.barh(range(len(words)), means, xerr=stds, color=colors, alpha=0.7)

            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=9)
            ax.set_xlabel("Attribution Score", fontsize=10)
            ax.set_title(
                f"{model_key.capitalize()} Model - {label_key.capitalize()} News",
                fontsize=12,
                fontweight="bold",
            )
            ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
            ax.grid(axis="x", alpha=0.3)

            # Invert y-axis so most important is on top
            ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved word importance plot to: {save_path}")


def plot_attribution_heatmap(attributions: Dict, save_path: str, n_samples: int = 10):
    """
    Create heatmap visualization of attributions for sample texts.

    Args:
        attributions: Attribution dictionary
        save_path: Path to save the plot
        n_samples: Number of samples to visualize
    """
    # Select samples (mix of real and fake)
    labels = attributions["labels"]
    real_indices = np.where(labels == 0)[0][: n_samples // 2]
    fake_indices = np.where(labels == 1)[0][: n_samples // 2]
    selected_indices = np.concatenate([real_indices, fake_indices])

    if len(selected_indices) == 0:
        print("No samples to visualize")
        return

    fig, axes = plt.subplots(
        len(selected_indices), 1, figsize=(16, 2 * len(selected_indices))
    )
    if len(selected_indices) == 1:
        axes = [axes]

    for idx, sample_idx in enumerate(selected_indices):
        ax = axes[idx]

        tokens = attributions["tokens"][sample_idx]
        attr = attributions["attributions"][sample_idx]
        label = attributions["labels"][sample_idx]

        # Filter out padding
        non_pad = [i for i, t in enumerate(tokens) if t not in ["[PAD]", "<pad>"]]
        tokens_filtered = [tokens[i] for i in non_pad]
        attr_filtered = [attr[i] for i in non_pad]

        # Limit to reasonable length
        max_len = 30
        if len(tokens_filtered) > max_len:
            tokens_filtered = tokens_filtered[:max_len]
            attr_filtered = attr_filtered[:max_len]

        # Clean subword tokens
        tokens_filtered = [
            t.replace("##", "").replace("Ġ", "") for t in tokens_filtered
        ]

        # Create heatmap
        attr_normalized = np.array(attr_filtered).reshape(1, -1)
        im = ax.imshow(
            attr_normalized, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.1
        )

        # Set ticks
        ax.set_xticks(range(len(tokens_filtered)))
        ax.set_xticklabels(tokens_filtered, rotation=45, ha="right", fontsize=8)
        ax.set_yticks([])

        label_str = "Fake" if label == 1 else "Real"
        ax.set_ylabel(label_str, fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved attribution heatmap to: {save_path}")


def compare_attributions(
    clean_importance: Dict, poisoned_importance: Dict, save_path: str
):
    """
    Compare word importance between clean and poisoned models.

    Args:
        clean_importance: Word importance from clean model
        poisoned_importance: Word importance from poisoned model
        save_path: Path to save comparison results
    """
    # Find words that changed importance significantly
    changes = {"real": {}, "fake": {}}

    for label in ["real", "fake"]:
        clean_words = set(clean_importance["clean"][label].keys())
        poisoned_words = set(poisoned_importance["poisoned"][label].keys())

        # Common words
        common = clean_words & poisoned_words

        for word in common:
            clean_score = clean_importance["clean"][label][word]["mean"]
            poisoned_score = poisoned_importance["poisoned"][label][word]["mean"]

            change = poisoned_score - clean_score
            if abs(change) > 0.01:  # Significant change threshold
                changes[label][word] = {
                    "clean": clean_score,
                    "poisoned": poisoned_score,
                    "change": change,
                }

        # New important words in poisoned
        new_words = poisoned_words - clean_words
        for word in new_words:
            if poisoned_importance["poisoned"][label][word]["count"] >= 3:
                changes[label][word] = {
                    "clean": 0.0,
                    "poisoned": poisoned_importance["poisoned"][label][word]["mean"],
                    "change": poisoned_importance["poisoned"][label][word]["mean"],
                }

    # Visualize changes
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for i, label in enumerate(["real", "fake"]):
        ax = axes[i]

        if not changes[label]:
            ax.text(0.5, 0.5, "No significant changes", ha="center", va="center")
            ax.set_title(f"{label.capitalize()} News")
            continue

        # Sort by absolute change
        sorted_changes = sorted(
            changes[label].items(), key=lambda x: abs(x[1]["change"]), reverse=True
        )[:20]

        words = [item[0] for item in sorted_changes]
        clean_scores = [item[1]["clean"] for item in sorted_changes]
        poisoned_scores = [item[1]["poisoned"] for item in sorted_changes]

        x = np.arange(len(words))
        width = 0.35

        ax.barh(
            x - width / 2, clean_scores, width, label="Clean", alpha=0.7, color="blue"
        )
        ax.barh(
            x + width / 2,
            poisoned_scores,
            width,
            label="Poisoned",
            alpha=0.7,
            color="red",
        )

        ax.set_yticks(x)
        ax.set_yticklabels(words, fontsize=9)
        ax.set_xlabel("Attribution Score", fontsize=10)
        ax.set_title(
            f"{label.capitalize()} News - Attribution Changes",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend()
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved attribution comparison to: {save_path}")
