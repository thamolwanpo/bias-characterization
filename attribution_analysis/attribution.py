"""
Axiomatic Attribution Analysis for News Recommendation Models.

Implements Integrated Gradients to analyze which words affect model recommendations
for real vs fake news classification on both clean and poisoned models.

Reference: Axiomatic Attribution for Deep Networks (Sundararajan et al., 2017)
https://arxiv.org/abs/1703.01365

Supported Architectures:
- NRMS: Title-only attribution analysis
- NAML: Multi-view attribution analysis (title + body text)
  * Separate attributions computed for title and body views
  * Handles NAML's multi-view encoder architecture
  * Can analyze both views independently

Word-Level Attributions:
- Token-level attributions are grouped into word-level attributions
- Subword tokens (BERT ##, RoBERTa Ġ) are merged into complete words
- Attribution scores are averaged per word: mean(token_attributions)
- Special tokens are filtered out during grouping

GPU Optimizations:
- Batched Integrated Gradients: Process multiple samples in parallel (10-20x speedup)
- History embedding caching: Pre-compute user embeddings once per batch
- Efficient gradient operations: Removed redundant .clone() calls
- Memory management: Periodic cache clearing and GPU monitoring
- Single GPU transfer: Entire batch moved to GPU at once instead of per-sample

Expected Performance:
- GPU utilization: 3-5GB+ (up from 1.5GB)
- Speed: ~2-4 hours for 40k samples (down from 45+ hours)
- Batch size: Configurable via DataLoader (default from model config)
- NAML: ~2x slower than NRMS due to dual attribution computation (title + body)
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

# Common English stopwords to filter out
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "will",
    "with",
    "the",
    "this",
    "but",
    "they",
    "have",
    "had",
    "what",
    "when",
    "where",
    "who",
    "which",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
}

# Add parent directory for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

sys.path.insert(
    0,
    os.path.abspath("/home/thamo/PhD/bias-characterization/plm4newsrs/src/models"),
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
                self.model.news_encoder, "lm"
            ):
                if hasattr(self.model.news_encoder, "bert"):
                    embedding_layer = self.model.news_encoder.bert.embeddings
                else:
                    embedding_layer = self.model.news_encoder.lm.embeddings
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

            elif hasattr(news_encoder, "lm"):
                # Bypass embeddings, go directly to encoder
                encoder_output = news_encoder.lm.encoder(
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


def group_tokens_to_words(
    tokens: List[str], attributions: np.ndarray
) -> Tuple[List[str], np.ndarray]:
    """
    Group subword tokens into words and average their attributions.

    Args:
        tokens: List of subword tokens (e.g., ["The", "play", "##ing", "field"])
        attributions: Attribution scores for each token [n_tokens]

    Returns:
        words: List of grouped words (e.g., ["The", "playing", "field"])
        word_attributions: Average attribution score for each word
    """
    if len(tokens) == 0:
        return [], np.array([])

    words = []
    word_attributions = []

    current_word = ""
    current_attrs = []

    for token, attr in zip(tokens, attributions):
        # Skip special tokens
        if token in ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "<pad>", "<s>", "</s>"]:
            # If we have a word in progress, save it
            if current_word:
                words.append(current_word)
                word_attributions.append(np.mean(current_attrs))
                current_word = ""
                current_attrs = []
            continue

        # Check if this is a continuation token (BERT-style)
        if token.startswith("##"):
            # Continuation of previous word
            current_word += token[2:]  # Remove "##" prefix
            current_attrs.append(attr)
        # Check if this is a continuation token (RoBERTa/GPT-style)
        elif token.startswith("Ġ"):
            # New word (Ġ indicates word boundary)
            if current_word:
                words.append(current_word)
                word_attributions.append(np.mean(current_attrs))
            current_word = token[1:]  # Remove "Ġ" prefix
            current_attrs = [attr]
        else:
            # For other tokens, check if we should start a new word
            # Simple heuristic: if previous word exists and this doesn't start with ##, it's a new word
            if current_word:
                words.append(current_word)
                word_attributions.append(np.mean(current_attrs))
            current_word = token
            current_attrs = [attr]

    # Don't forget the last word
    if current_word:
        words.append(current_word)
        word_attributions.append(np.mean(current_attrs))

    return words, np.array(word_attributions)


def compute_attributions_transformer_naml(
    model,
    candidate_title_ids: torch.Tensor,
    candidate_title_mask: torch.Tensor,
    candidate_body_ids: torch.Tensor,
    candidate_body_mask: torch.Tensor,
    history_title_ids: torch.Tensor,
    history_title_mask: torch.Tensor,
    history_body_ids: torch.Tensor,
    history_body_mask: torch.Tensor,
    target_candidate_idx: int = 0,
    n_steps: int = 50,
    user_emb_cache: Optional[torch.Tensor] = None,
    attribution_target: str = "title",  # "title" or "body"
    return_completeness: bool = True,
) -> Tuple[torch.Tensor, Optional[Dict]]:
    """
    Compute attributions for NAML transformer-based models.

    NAML uses multi-view learning with separate encoders for title and body.
    This function computes attributions for either title or body text.

    Args:
        model: Full NAML recommendation model
        candidate_title_ids: Candidate title token IDs [batch_size, n_candidates, seq_len]
        candidate_title_mask: Candidate title attention mask [batch_size, n_candidates, seq_len]
        candidate_body_ids: Candidate body token IDs [batch_size, n_candidates, seq_len]
        candidate_body_mask: Candidate body attention mask [batch_size, n_candidates, seq_len]
        history_title_ids: History title token IDs [batch_size, history_len, seq_len]
        history_title_mask: History title attention mask [batch_size, history_len, seq_len]
        history_body_ids: History body token IDs [batch_size, history_len, seq_len]
        history_body_mask: History body attention mask [batch_size, history_len, seq_len]
        target_candidate_idx: Which candidate to compute attributions for
        n_steps: Number of integration steps
        user_emb_cache: Pre-computed user embeddings [batch_size, embed_dim] (optional)
        attribution_target: Which view to compute attributions for ("title" or "body")
        return_completeness: Whether to return completeness check metrics

    Returns:
        attributions: Attribution scores [batch_size, seq_len]
        completeness: Completeness check metrics (if return_completeness=True)
    """
    device = candidate_title_ids.device
    news_encoder = model.news_encoder
    user_encoder = model.user_encoder

    # Get the embedding layer for the target view
    embedding_layer = None

    if hasattr(news_encoder, "text_encoders"):
        # NAML has separate text encoders for title and body
        if attribution_target == "title" and "title" in news_encoder.text_encoders:
            text_encoder = news_encoder.text_encoders["title"]
        elif attribution_target == "body" and "text" in news_encoder.text_encoders:
            text_encoder = news_encoder.text_encoders["text"]
        else:
            raise ValueError(f"Cannot find {attribution_target} encoder in NAML model")

        # Get embedding layer from text encoder
        if hasattr(text_encoder, "bert"):
            embedding_layer = text_encoder.bert.embeddings
        elif hasattr(text_encoder, "lm"):
            embedding_layer = text_encoder.lm.embeddings

    if embedding_layer is None:
        raise ValueError(
            f"Cannot find embedding layer for {attribution_target} in NAML model"
        )

    batch_size, num_candidates, seq_len = candidate_title_ids.shape

    # Select which inputs to use based on attribution target
    if attribution_target == "title":
        target_ids = candidate_title_ids[:, target_candidate_idx, :]
        target_mask = candidate_title_mask[:, target_candidate_idx, :]
    else:  # body
        target_ids = candidate_body_ids[:, target_candidate_idx, :]
        target_mask = candidate_body_mask[:, target_candidate_idx, :]

    baseline_ids = torch.zeros_like(target_ids)  # PAD tokens

    with torch.no_grad():
        # Get embeddings
        input_embeddings = embedding_layer(target_ids)
        baseline_embeddings = embedding_layer(baseline_ids)

        # Compute or use cached user embeddings
        if user_emb_cache is None:
            history_len = history_title_ids.shape[1]
            _, _, title_seq_len = history_title_ids.shape
            _, _, body_seq_len = history_body_ids.shape

            # Encode history using NAML's multi-view encoding
            history_embs = encode_naml_news_batch(
                news_encoder,
                history_title_ids.view(batch_size * history_len, title_seq_len),
                history_title_mask.view(batch_size * history_len, title_seq_len),
                history_body_ids.view(batch_size * history_len, body_seq_len),
                history_body_mask.view(batch_size * history_len, body_seq_len),
            ).view(batch_size, history_len, -1)

            user_emb = user_encoder(history_embs)
        else:
            user_emb = user_emb_cache

    # Accumulate gradients
    accumulated_grads = torch.zeros_like(input_embeddings)

    for step in range(n_steps):
        alpha = (step + 1) / n_steps
        interpolated = baseline_embeddings + alpha * (
            input_embeddings - baseline_embeddings
        )
        interpolated.requires_grad_(True)

        # Encode candidate using NAML with interpolated embeddings for target view
        if attribution_target == "title":
            candidate_emb = encode_naml_news_from_embeddings(
                news_encoder,
                title_embeddings=interpolated,
                title_mask=target_mask,
                body_ids=candidate_body_ids[:, target_candidate_idx, :],
                body_mask=candidate_body_mask[:, target_candidate_idx, :],
            )
        else:  # body
            candidate_emb = encode_naml_news_from_embeddings(
                news_encoder,
                title_ids=candidate_title_ids[:, target_candidate_idx, :],
                title_mask=candidate_title_mask[:, target_candidate_idx, :],
                body_embeddings=interpolated,
                body_mask=target_mask,
            )

        # Compute score
        score = torch.sum(candidate_emb * user_emb, dim=-1)

        # Backward pass
        score.sum().backward()

        # Accumulate gradients and properly clean up
        if interpolated.grad is not None:
            accumulated_grads += interpolated.grad.detach().clone()
            interpolated.grad = None

        # Explicitly delete tensors to free memory
        del candidate_emb, score, interpolated

    # Average and multiply by difference
    avg_grads = accumulated_grads / n_steps
    attributions = avg_grads * (input_embeddings - baseline_embeddings)

    # Sum over embedding dimension
    token_attributions = attributions.sum(dim=-1)

    # Compute completeness check if requested
    completeness_metrics = None
    if return_completeness:
        with torch.no_grad():
            # Compute input score (F(x))
            if attribution_target == "title":
                input_candidate_emb = encode_naml_news_from_embeddings(
                    news_encoder,
                    title_embeddings=input_embeddings,
                    title_mask=target_mask,
                    body_ids=candidate_body_ids[:, target_candidate_idx, :],
                    body_mask=candidate_body_mask[:, target_candidate_idx, :],
                )
            else:  # body
                input_candidate_emb = encode_naml_news_from_embeddings(
                    news_encoder,
                    title_ids=candidate_title_ids[:, target_candidate_idx, :],
                    title_mask=candidate_title_mask[:, target_candidate_idx, :],
                    body_embeddings=input_embeddings,
                    body_mask=target_mask,
                )
            input_score = torch.sum(input_candidate_emb * user_emb, dim=-1)

            # Compute baseline score (F(baseline))
            if attribution_target == "title":
                baseline_candidate_emb = encode_naml_news_from_embeddings(
                    news_encoder,
                    title_embeddings=baseline_embeddings,
                    title_mask=target_mask,
                    body_ids=candidate_body_ids[:, target_candidate_idx, :],
                    body_mask=candidate_body_mask[:, target_candidate_idx, :],
                )
            else:  # body
                baseline_candidate_emb = encode_naml_news_from_embeddings(
                    news_encoder,
                    title_ids=candidate_title_ids[:, target_candidate_idx, :],
                    title_mask=candidate_title_mask[:, target_candidate_idx, :],
                    body_embeddings=baseline_embeddings,
                    body_mask=target_mask,
                )
            baseline_score = torch.sum(baseline_candidate_emb * user_emb, dim=-1)

            # Sum attributions over all tokens
            attribution_sum = token_attributions.sum(dim=-1)  # [batch_size]

            # Check completeness
            completeness_metrics = compute_completeness_check(
                attribution_sum, input_score, baseline_score
            )

    return token_attributions, completeness_metrics


def encode_naml_news_batch(news_encoder, title_ids, title_mask, body_ids, body_mask):
    """Encode news using NAML's multi-view architecture from token IDs."""
    return news_encoder(
        title_input_ids=title_ids,
        title_attention_mask=title_mask,
        body_input_ids=body_ids,
        body_attention_mask=body_mask,
    )


def encode_naml_news_from_embeddings(
    news_encoder,
    title_ids=None,
    title_embeddings=None,
    title_mask=None,
    body_ids=None,
    body_embeddings=None,
    body_mask=None,
):
    """
    Encode news using NAML with embeddings for one view and IDs for the other.

    This allows us to compute gradients for one view while keeping the other fixed.
    """
    # Get text encoders
    title_encoder = news_encoder.text_encoders["title"]
    body_encoder = news_encoder.text_encoders["text"]

    # Encode title view
    if title_embeddings is not None:
        # Use interpolated embeddings for title
        title_emb = encode_text_from_embeddings(
            title_encoder, title_embeddings, title_mask
        )
    elif title_ids is not None:
        # Use normal encoding for title
        title_emb = title_encoder(
            torch.stack([title_ids, title_mask], dim=1)
            if title_ids.dim() == 2
            else title_ids
        )
    else:
        raise ValueError("Either title_ids or title_embeddings must be provided")

    # Encode body view
    if body_embeddings is not None:
        # Use interpolated embeddings for body
        body_emb = encode_text_from_embeddings(body_encoder, body_embeddings, body_mask)
    elif body_ids is not None:
        # Use normal encoding for body
        body_emb = body_encoder(
            torch.stack([body_ids, body_mask], dim=1)
            if body_ids.dim() == 2
            else body_ids
        )
    else:
        raise ValueError("Either body_ids or body_embeddings must be provided")

    # Combine views using NAML's multi-view attention
    if (
        hasattr(news_encoder, "final_attention")
        and news_encoder.final_attention is not None
    ):
        # Stack views and apply attention
        stacked_views = torch.stack([title_emb, body_emb], dim=1)  # [batch, 2, dim]
        news_emb, _ = news_encoder.final_attention(stacked_views)
    else:
        # Simple averaging if only one view
        news_emb = (title_emb + body_emb) / 2

    return news_emb


def encode_text_from_embeddings(text_encoder, embeddings, attention_mask):
    """Encode text from embeddings through a NAML text encoder."""
    # Get transformer backend
    if hasattr(text_encoder, "bert"):
        transformer = text_encoder.bert
        encoder = transformer.encoder
    elif hasattr(text_encoder, "lm"):
        transformer = text_encoder.lm
        encoder = transformer.encoder
    else:
        raise ValueError("Cannot find transformer in text encoder")

    # Create extended attention mask
    batch_size, seq_length = attention_mask.shape
    extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = extended_attention_mask.to(dtype=embeddings.dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
        embeddings.dtype
    ).min

    # Pass through encoder
    encoder_output = encoder(embeddings, attention_mask=extended_attention_mask)

    if hasattr(encoder_output, "last_hidden_state"):
        sequence_output = encoder_output.last_hidden_state
    elif isinstance(encoder_output, tuple):
        sequence_output = encoder_output[0]
    else:
        sequence_output = encoder_output

    # Apply final layers (CNN + attention in NAML)
    if hasattr(text_encoder, "CNN"):
        # NAML's BERTTextEncoder uses pooler then CNN
        if hasattr(text_encoder, "pooler"):
            cls_token = sequence_output[:, 0]  # [batch, hidden_dim]
            pooled = text_encoder.pooler(cls_token)
            # CNN expects [batch, 1, 1, dim]
            convoluted = text_encoder.CNN(
                pooled.unsqueeze(1).unsqueeze(2).float()
            ).squeeze(dim=3)
        else:
            # Fallback: apply CNN directly
            convoluted = text_encoder.CNN(sequence_output.unsqueeze(1)).squeeze(dim=3)

        # Apply activation and dropout
        import torch.nn.functional as F

        activated = F.dropout(
            F.relu(convoluted),
            p=text_encoder.dropout_probability,
            training=text_encoder.training,
        )

        # Apply additive attention
        if hasattr(text_encoder, "additive_attention"):
            text_emb, _ = text_encoder.additive_attention(activated.transpose(1, 2))
        else:
            text_emb = activated.mean(dim=1)
    elif hasattr(text_encoder, "additive_attention"):
        # Apply additive attention directly
        text_emb, _ = text_encoder.additive_attention(sequence_output)
    else:
        # Use CLS token
        text_emb = sequence_output[:, 0, :]

    return text_emb


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

    # Print GPU info if using CUDA
    if device != "cpu" and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(
            f"Initial GPU Memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB"
        )

    use_glove = "glove" in model_config.model_name.lower()
    architecture = model_config.architecture
    is_naml = architecture == "naml"

    # Storage
    all_attributions = []
    all_tokens = []
    all_labels = []
    all_scores = []
    all_predictions = []

    # Additional storage for NAML body attributions
    all_body_attributions = []
    all_body_tokens = []

    # Storage for completeness metrics
    all_completeness_metrics = {
        "expected_diff": [],
        "actual_sum": [],
        "abs_error": [],
        "rel_error_percent": [],
    }
    # For NAML, track body completeness separately
    if is_naml:
        all_body_completeness_metrics = {
            "expected_diff": [],
            "actual_sum": [],
            "abs_error": [],
            "rel_error_percent": [],
        }

    print(f"Extracting attributions for {n_samples} samples...")
    print(f"  Model type: {'GloVe' if use_glove else 'Transformer'}")
    print(f"  Architecture: {architecture}")
    if is_naml:
        print(f"  NAML: Extracting attributions for both title and body")

    sample_count = 0
    for batch in tqdm(data_loader, desc="Processing batches"):
        if sample_count >= n_samples:
            break

        if batch is None:
            continue

        batch_size = len(batch.get("impression_data", []))
        # Limit batch size to remaining samples
        effective_batch_size = min(batch_size, n_samples - sample_count)

        # Get labels
        if "impression_data" in batch:
            for i in range(effective_batch_size):
                impression = batch["impression_data"][i]
                is_fake = impression[0][3]  # First candidate's label
                all_labels.append(is_fake)

        try:
            if use_glove:
                # GloVe models - still process individually due to text-based nature
                for i in range(effective_batch_size):
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

                    # Ensure attributions match tokens length
                    if len(attributions) > len(tokens):
                        attributions = attributions[: len(tokens)]
                    elif len(attributions) < len(tokens):
                        # Pad with zeros if needed
                        padding = torch.zeros(
                            len(tokens) - len(attributions), device=device
                        )
                        attributions = torch.cat([attributions, padding])

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
                # Transformer models - BATCHED PROCESSING
                # Move entire batch to device at once
                candidate_title_ids = batch["candidate_title_input_ids"][
                    :effective_batch_size
                ].to(device)
                candidate_title_mask = batch["candidate_title_attention_mask"][
                    :effective_batch_size
                ].to(device)
                history_title_ids = batch["history_title_input_ids"][
                    :effective_batch_size
                ].to(device)
                history_title_mask = batch["history_title_attention_mask"][
                    :effective_batch_size
                ].to(device)

                # NAML: also get body text
                if is_naml:
                    candidate_body_ids = batch["candidate_input_ids"][
                        :effective_batch_size
                    ].to(device)
                    candidate_body_mask = batch["candidate_attention_mask"][
                        :effective_batch_size
                    ].to(device)
                    history_body_ids = batch["history_input_ids"][
                        :effective_batch_size
                    ].to(device)
                    history_body_mask = batch["history_attention_mask"][
                        :effective_batch_size
                    ].to(device)

                # Pre-compute user embeddings once for the entire batch (caching)
                with torch.no_grad():
                    batch_size_actual = candidate_title_ids.shape[0]
                    history_len = history_title_ids.shape[1]
                    seq_len = history_title_ids.shape[2]

                    if is_naml:
                        # NAML: encode history with both title and body
                        body_seq_len = history_body_ids.shape[2]
                        history_embs = encode_naml_news_batch(
                            model.news_encoder,
                            history_title_ids.view(
                                batch_size_actual * history_len, seq_len
                            ),
                            history_title_mask.view(
                                batch_size_actual * history_len, seq_len
                            ),
                            history_body_ids.view(
                                batch_size_actual * history_len, body_seq_len
                            ),
                            history_body_mask.view(
                                batch_size_actual * history_len, body_seq_len
                            ),
                        ).view(batch_size_actual, history_len, -1)
                    else:
                        # NRMS: encode history with title only
                        history_flat_ids = history_title_ids.view(
                            batch_size_actual * history_len, seq_len
                        )
                        history_flat_mask = history_title_mask.view(
                            batch_size_actual * history_len, seq_len
                        )

                        history_embs = encode_transformer_news(
                            model.news_encoder, history_flat_ids, history_flat_mask
                        ).view(batch_size_actual, history_len, -1)

                    user_embs = model.user_encoder(
                        history_embs
                    )  # [batch_size, embed_dim]

                # Compute attributions for entire batch at once
                with torch.enable_grad():
                    if is_naml:
                        # NAML: compute attributions for both title and body
                        # Title attributions
                        (
                            title_attributions_batch,
                            title_completeness,
                        ) = compute_attributions_transformer_naml(
                            model,
                            candidate_title_ids,
                            candidate_title_mask,
                            candidate_body_ids,
                            candidate_body_mask,
                            history_title_ids,
                            history_title_mask,
                            history_body_ids,
                            history_body_mask,
                            target_candidate_idx=0,
                            n_steps=n_steps,
                            user_emb_cache=user_embs,
                            attribution_target="title",
                            return_completeness=True,
                        )  # [batch_size, seq_len]

                        # Body attributions
                        (
                            body_attributions_batch,
                            body_completeness,
                        ) = compute_attributions_transformer_naml(
                            model,
                            candidate_title_ids,
                            candidate_title_mask,
                            candidate_body_ids,
                            candidate_body_mask,
                            history_title_ids,
                            history_title_mask,
                            history_body_ids,
                            history_body_mask,
                            target_candidate_idx=0,
                            n_steps=n_steps,
                            user_emb_cache=user_embs,
                            attribution_target="body",
                            return_completeness=True,
                        )  # [batch_size, seq_len]

                        # Collect completeness metrics for title
                        if title_completeness:
                            all_completeness_metrics["expected_diff"].extend(
                                title_completeness["expected_diff"]
                            )
                            all_completeness_metrics["actual_sum"].extend(
                                title_completeness["actual_sum"]
                            )
                            all_completeness_metrics["abs_error"].extend(
                                title_completeness["abs_error"]
                            )
                            all_completeness_metrics["rel_error_percent"].extend(
                                title_completeness["rel_error_percent"]
                            )

                        # Collect completeness metrics for body
                        if body_completeness:
                            all_body_completeness_metrics["expected_diff"].extend(
                                body_completeness["expected_diff"]
                            )
                            all_body_completeness_metrics["actual_sum"].extend(
                                body_completeness["actual_sum"]
                            )
                            all_body_completeness_metrics["abs_error"].extend(
                                body_completeness["abs_error"]
                            )
                            all_body_completeness_metrics["rel_error_percent"].extend(
                                body_completeness["rel_error_percent"]
                            )
                    else:
                        # NRMS: compute attributions for title only
                        (
                            title_attributions_batch,
                            title_completeness,
                        ) = compute_attributions_transformer(
                            model,
                            candidate_title_ids,
                            candidate_title_mask,
                            history_title_ids,
                            history_title_mask,
                            target_candidate_idx=0,
                            n_steps=n_steps,
                            user_emb_cache=user_embs,  # Pass cached user embeddings
                            return_completeness=True,
                        )  # [batch_size, seq_len]

                        # Collect completeness metrics
                        if title_completeness:
                            all_completeness_metrics["expected_diff"].extend(
                                title_completeness["expected_diff"]
                            )
                            all_completeness_metrics["actual_sum"].extend(
                                title_completeness["actual_sum"]
                            )
                            all_completeness_metrics["abs_error"].extend(
                                title_completeness["abs_error"]
                            )
                            all_completeness_metrics["rel_error_percent"].extend(
                                title_completeness["rel_error_percent"]
                            )

                # Get prediction scores for entire batch
                with torch.no_grad():
                    batch_dict = {
                        "candidate_title_input_ids": candidate_title_ids,
                        "candidate_title_attention_mask": candidate_title_mask,
                        "history_title_input_ids": history_title_ids,
                        "history_title_attention_mask": history_title_mask,
                    }
                    if is_naml:
                        batch_dict["candidate_input_ids"] = candidate_body_ids
                        batch_dict["candidate_attention_mask"] = candidate_body_mask
                        batch_dict["history_input_ids"] = history_body_ids
                        batch_dict["history_attention_mask"] = history_body_mask

                    scores_batch = model(batch_dict)  # [batch_size, n_candidates]
                    predictions_batch = scores_batch.argmax(dim=1)  # [batch_size]
                    scores_batch_first = scores_batch[
                        :, 0
                    ]  # [batch_size] - first candidate scores

                # Get tokenizer once
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)

                # Store results for each sample in batch
                for i in range(batch_size_actual):
                    # Title: Get first candidate tokens and attributions
                    title_input_ids = candidate_title_ids[i, 0, :]
                    title_tokens = tokenizer.convert_ids_to_tokens(
                        title_input_ids.cpu().numpy()
                    )
                    title_attributions = (
                        title_attributions_batch[i].cpu().numpy()
                    )  # [seq_len]

                    # Group tokens into words and average attributions
                    title_words, title_word_attrs = group_tokens_to_words(
                        title_tokens, title_attributions
                    )

                    all_tokens.append(title_words)
                    all_attributions.append(title_word_attrs)

                    # Body: For NAML, also process body attributions
                    if is_naml:
                        body_input_ids = candidate_body_ids[i, 0, :]
                        body_tokens = tokenizer.convert_ids_to_tokens(
                            body_input_ids.cpu().numpy()
                        )
                        body_attributions = (
                            body_attributions_batch[i].cpu().numpy()
                        )  # [seq_len]

                        # Group tokens into words and average attributions
                        body_words, body_word_attrs = group_tokens_to_words(
                            body_tokens, body_attributions
                        )

                        all_body_tokens.append(body_words)
                        all_body_attributions.append(body_word_attrs)

                    # Get prediction and score
                    all_predictions.append(predictions_batch[i].item())
                    all_scores.append(scores_batch_first[i].item())

        except Exception as e:
            print(
                f"\nWarning: Failed to compute attributions for batch at sample {sample_count}: {e}"
            )
            import traceback

            traceback.print_exc()

            # Add dummy data for failed batch
            for i in range(effective_batch_size):
                all_attributions.append(np.zeros(10))
                all_tokens.append(["[ERROR]"] * 10)
                if is_naml:
                    all_body_attributions.append(np.zeros(10))
                    all_body_tokens.append(["[ERROR]"] * 10)
                all_predictions.append(0)
                all_scores.append(0.0)

        sample_count += effective_batch_size

        # Periodic GPU memory cleanup
        if device != "cpu" and torch.cuda.is_available() and sample_count % 100 == 0:
            try:
                torch.cuda.empty_cache()
                if sample_count % 500 == 0:  # Report every 500 samples
                    print(
                        f"  GPU Memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB"
                    )
            except RuntimeError as e:
                print(f"\nWarning: Error during GPU cache cleanup: {e}")
                # Continue processing despite cache cleanup error

    print(f"\nExtracted attributions for {len(all_attributions)} samples")
    if is_naml:
        print(f"  Title attributions: {len(all_attributions)} samples")
        print(f"  Body attributions: {len(all_body_attributions)} samples")

    # Print completeness check results
    print("\n" + "=" * 75)
    print("INTEGRATED GRADIENTS COMPLETENESS CHECK (Proposition 1)")
    print("=" * 75)
    print("Verifies that: ∑ Attribution_i ≈ F(x) - F(baseline)")
    print(
        "Recommended: Attributions should sum to within 5% of score difference\n"
    )

    # Title completeness metrics
    if all_completeness_metrics["rel_error_percent"]:
        rel_errors = np.array(all_completeness_metrics["rel_error_percent"])
        abs_errors = np.array(all_completeness_metrics["abs_error"])
        expected_diffs = np.array(all_completeness_metrics["expected_diff"])
        actual_sums = np.array(all_completeness_metrics["actual_sum"])

        view_name = "Title" if is_naml else ""
        print(f"{view_name} Attribution Completeness:")
        print(f"  Mean relative error: {rel_errors.mean():.2f}%")
        print(f"  Median relative error: {np.median(rel_errors):.2f}%")
        print(f"  Max relative error: {rel_errors.max():.2f}%")
        print(f"  Samples within 5% error: {(rel_errors <= 5).sum()}/{len(rel_errors)} ({(rel_errors <= 5).mean() * 100:.1f}%)")
        print(f"  Mean absolute error: {abs_errors.mean():.4f}")
        print(
            f"  Mean expected diff [F(x) - F(baseline)]: {expected_diffs.mean():.4f}"
        )
        print(f"  Mean actual sum [∑ Attribution_i]: {actual_sums.mean():.4f}")

        # Warn if errors are too high
        if rel_errors.mean() > 5:
            print(
                f"\n  ⚠️  WARNING: Mean relative error ({rel_errors.mean():.2f}%) exceeds 5%"
            )
            print(
                f"  Consider increasing n_steps (currently {n_steps}) for better approximation"
            )
        else:
            print(f"\n  ✓ Completeness check passed (mean error: {rel_errors.mean():.2f}%)")

    # Body completeness metrics (NAML only)
    if is_naml and all_body_completeness_metrics["rel_error_percent"]:
        rel_errors = np.array(all_body_completeness_metrics["rel_error_percent"])
        abs_errors = np.array(all_body_completeness_metrics["abs_error"])
        expected_diffs = np.array(all_body_completeness_metrics["expected_diff"])
        actual_sums = np.array(all_body_completeness_metrics["actual_sum"])

        print(f"\nBody Attribution Completeness:")
        print(f"  Mean relative error: {rel_errors.mean():.2f}%")
        print(f"  Median relative error: {np.median(rel_errors):.2f}%")
        print(f"  Max relative error: {rel_errors.max():.2f}%")
        print(f"  Samples within 5% error: {(rel_errors <= 5).sum()}/{len(rel_errors)} ({(rel_errors <= 5).mean() * 100:.1f}%)")
        print(f"  Mean absolute error: {abs_errors.mean():.4f}")
        print(
            f"  Mean expected diff [F(x) - F(baseline)]: {expected_diffs.mean():.4f}"
        )
        print(f"  Mean actual sum [∑ Attribution_i]: {actual_sums.mean():.4f}")

        # Warn if errors are too high
        if rel_errors.mean() > 5:
            print(
                f"\n  ⚠️  WARNING: Mean relative error ({rel_errors.mean():.2f}%) exceeds 5%"
            )
            print(
                f"  Consider increasing n_steps (currently {n_steps}) for better approximation"
            )
        else:
            print(f"\n  ✓ Completeness check passed (mean error: {rel_errors.mean():.2f}%)")

    print("=" * 75)

    # Final cleanup
    if device != "cpu" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            print(
                f"\nFinal GPU Memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB"
            )
        except RuntimeError as e:
            print(f"\nWarning: Error during final GPU cache cleanup: {e}")

    result = {
        "attributions": all_attributions,
        "tokens": all_tokens,
        "labels": np.array(all_labels[: len(all_attributions)]),
        "scores": np.array(all_scores),
        "predictions": np.array(all_predictions),
        "completeness_metrics": all_completeness_metrics,
    }

    # Add body attributions for NAML
    if is_naml:
        result["body_attributions"] = all_body_attributions
        result["body_tokens"] = all_body_tokens
        result["body_completeness_metrics"] = all_body_completeness_metrics

    return result


def compute_attributions_glove(
    model, candidate_text: str, history_texts: List[str], device, n_steps: int = 50
) -> torch.Tensor:
    """
    Compute attributions for GloVe-based models using occlusion-based approach.

    For GloVe models, we compute word-level attributions by measuring the impact
    of removing each word on the model's prediction score.

    Args:
        model: Full recommendation model
        candidate_text: Candidate news text
        history_texts: List of history news texts
        device: Device to run on
        n_steps: Number of integration steps (unused for occlusion, kept for API compatibility)

    Returns:
        attributions: Attribution scores for each word [n_words]
    """
    # Tokenize text (simple word splitting)
    words = candidate_text.split()
    n_words = len(words)

    if n_words == 0:
        return torch.zeros(1, device=device)

    news_encoder = model.news_encoder
    user_encoder = model.user_encoder

    # Device indicator for GloVe models
    device_indicator = torch.tensor([0], device=device)

    # Get user embedding (fixed during attribution)
    with torch.no_grad():
        # Encode history
        history_embs = []
        for hist_text in history_texts:
            hist_emb = news_encoder(input_ids=device_indicator, text_list=[hist_text])
            history_embs.append(hist_emb)

        if len(history_embs) > 0:
            history_embs = torch.stack(history_embs).unsqueeze(
                0
            )  # [1, history_len, dim]
            user_emb = user_encoder(history_embs).squeeze(0)  # [dim]
        else:
            # No history - use zero user embedding
            user_emb = torch.zeros(
                news_encoder.output_dim if hasattr(news_encoder, "output_dim") else 400,
                device=device,
            )

        # Get baseline score with full text
        candidate_emb_full = news_encoder(
            input_ids=device_indicator, text_list=[candidate_text]
        )
        baseline_score = (
            torch.matmul(candidate_emb_full, user_emb.unsqueeze(-1)).squeeze().item()
        )

        # Compute attribution for each word by occlusion
        word_attributions = torch.zeros(n_words, device=device)

        for i in range(n_words):
            # Create text with word i removed
            words_occluded = words[:i] + words[i + 1 :]
            text_occluded = " ".join(words_occluded) if words_occluded else ""

            # Get score without this word
            if text_occluded:
                candidate_emb_occluded = news_encoder(
                    input_ids=device_indicator, text_list=[text_occluded]
                )
                occluded_score = (
                    torch.matmul(candidate_emb_occluded, user_emb.unsqueeze(-1))
                    .squeeze()
                    .item()
                )
            else:
                # Empty text -> score of 0
                occluded_score = 0.0

            # Attribution is the difference when removing the word
            # Positive attribution means the word increases the score
            word_attributions[i] = baseline_score - occluded_score

    return word_attributions


def compute_completeness_check(
    attribution_sum: torch.Tensor,
    input_score: torch.Tensor,
    baseline_score: torch.Tensor,
) -> Dict[str, float]:
    """
    Check completeness of Integrated Gradients (Proposition 1).

    Verifies that: ∑ Attribution_i ≈ F(x) - F(baseline)

    Args:
        attribution_sum: Sum of all attributions [batch_size]
        input_score: Model score at input [batch_size]
        baseline_score: Model score at baseline [batch_size]

    Returns:
        Dictionary with completeness metrics per sample
    """
    # Expected difference
    expected_diff = input_score - baseline_score

    # Actual sum of attributions
    actual_sum = attribution_sum

    # Absolute error
    abs_error = torch.abs(actual_sum - expected_diff)

    # Relative error (percentage)
    # Avoid division by zero
    rel_error = abs_error / (torch.abs(expected_diff) + 1e-10) * 100

    return {
        "expected_diff": expected_diff.cpu().numpy(),
        "actual_sum": actual_sum.cpu().numpy(),
        "abs_error": abs_error.cpu().numpy(),
        "rel_error_percent": rel_error.cpu().numpy(),
    }


def compute_attributions_transformer(
    model,
    candidate_title_ids: torch.Tensor,
    candidate_title_mask: torch.Tensor,
    history_title_ids: torch.Tensor,
    history_title_mask: torch.Tensor,
    target_candidate_idx: int = 0,
    n_steps: int = 50,
    user_emb_cache: Optional[torch.Tensor] = None,
    return_completeness: bool = True,
) -> Tuple[torch.Tensor, Optional[Dict]]:
    """
    Compute attributions for transformer-based models through full architecture.

    Args:
        model: Full recommendation model
        candidate_title_ids: Candidate token IDs [batch_size, n_candidates, seq_len]
        candidate_title_mask: Candidate attention mask [batch_size, n_candidates, seq_len]
        history_title_ids: History token IDs [batch_size, history_len, seq_len]
        history_title_mask: History attention mask [batch_size, history_len, seq_len]
        target_candidate_idx: Which candidate to compute attributions for
        n_steps: Number of integration steps
        user_emb_cache: Pre-computed user embeddings [batch_size, embed_dim] (optional)
        return_completeness: Whether to return completeness check metrics

    Returns:
        attributions: Attribution scores [batch_size, seq_len]
        completeness: Completeness check metrics (if return_completeness=True)
    """
    device = candidate_title_ids.device
    news_encoder = model.news_encoder
    user_encoder = model.user_encoder

    # Get the embedding layer
    # Try different possible attribute names for various transformer architectures
    embedding_layer = None

    if hasattr(news_encoder, "bert"):
        # BERT-based models
        embedding_layer = news_encoder.bert.embeddings
    elif hasattr(news_encoder, "lm"):
        embedding_layer = news_encoder.lm.embeddings
    elif hasattr(news_encoder, "embeddings"):
        # Direct embeddings attribute
        embedding_layer = news_encoder.embeddings
    elif hasattr(news_encoder, "encoder") and hasattr(
        news_encoder.encoder, "embeddings"
    ):
        # Nested encoder structure
        embedding_layer = news_encoder.encoder.embeddings

    if embedding_layer is None:
        # Debug information
        print(f"\nDEBUG: News encoder type: {type(news_encoder)}")
        print(f"DEBUG: News encoder attributes: {dir(news_encoder)}")
        raise ValueError(
            f"Cannot find embedding layer in transformer model. "
            f"News encoder type: {type(news_encoder).__name__}. "
            f"Available attributes: {[attr for attr in dir(news_encoder) if not attr.startswith('_')]}"
        )

    # Support both single sample [1, n_candidates, seq_len] and batched [batch_size, n_candidates, seq_len]
    batch_size, num_candidates, seq_len = candidate_title_ids.shape

    # Get input and baseline embeddings for target candidate
    target_ids = candidate_title_ids[
        :, target_candidate_idx, :
    ]  # [batch_size, seq_len]
    target_mask = candidate_title_mask[
        :, target_candidate_idx, :
    ]  # [batch_size, seq_len]

    baseline_ids = torch.zeros_like(target_ids)  # PAD tokens

    with torch.no_grad():
        # Get embeddings - now batched
        input_embeddings = embedding_layer(
            target_ids
        )  # [batch_size, seq_len, embed_dim]
        baseline_embeddings = embedding_layer(baseline_ids)

        # Compute or use cached user embeddings
        if user_emb_cache is None:
            # Encode history to get user embedding (fixed during attribution)
            history_len = history_title_ids.shape[1]
            history_flat_ids = history_title_ids.view(batch_size * history_len, seq_len)
            history_flat_mask = history_title_mask.view(
                batch_size * history_len, seq_len
            )

            history_embs = encode_transformer_news(
                news_encoder, history_flat_ids, history_flat_mask
            ).view(batch_size, history_len, -1)

            user_emb = user_encoder(history_embs)  # [batch_size, embed_dim]
        else:
            user_emb = user_emb_cache  # [batch_size, embed_dim]

    # Accumulate gradients - now batched
    accumulated_grads = torch.zeros_like(input_embeddings)

    for step in range(n_steps):
        alpha = (step + 1) / n_steps
        interpolated = baseline_embeddings + alpha * (
            input_embeddings - baseline_embeddings
        )
        interpolated.requires_grad_(True)

        # Encode interpolated candidate - batched
        candidate_emb = encode_transformer_news_from_embeddings(
            news_encoder, interpolated, target_mask
        )  # [batch_size, embed_dim]

        # Compute score: batched dot product with user embedding
        # score = (candidate_emb * user_emb).sum(dim=-1)  # [batch_size]
        score = torch.sum(candidate_emb * user_emb, dim=-1)  # [batch_size]

        # Backward pass - sum to scalar for backward
        score.sum().backward()

        # Accumulate gradients and properly clean up
        if interpolated.grad is not None:
            accumulated_grads += interpolated.grad.detach().clone()
            interpolated.grad = None

        # Explicitly delete tensors to free memory
        del candidate_emb, score, interpolated

    # Average and multiply by difference
    avg_grads = accumulated_grads / n_steps
    attributions = avg_grads * (input_embeddings - baseline_embeddings)

    # Sum over embedding dimension
    token_attributions = attributions.sum(dim=-1)  # [batch_size, seq_len]

    # Compute completeness check if requested
    completeness_metrics = None
    if return_completeness:
        with torch.no_grad():
            # Compute input score (F(x))
            input_candidate_emb = encode_transformer_news_from_embeddings(
                news_encoder, input_embeddings, target_mask
            )
            input_score = torch.sum(input_candidate_emb * user_emb, dim=-1)

            # Compute baseline score (F(baseline))
            baseline_candidate_emb = encode_transformer_news_from_embeddings(
                news_encoder, baseline_embeddings, target_mask
            )
            baseline_score = torch.sum(baseline_candidate_emb * user_emb, dim=-1)

            # Sum attributions over all tokens
            attribution_sum = token_attributions.sum(dim=-1)  # [batch_size]

            # Check completeness
            completeness_metrics = compute_completeness_check(
                attribution_sum, input_score, baseline_score
            )

    return token_attributions, completeness_metrics


def encode_transformer_news(news_encoder, input_ids, attention_mask):
    """Encode news from token IDs through transformer news encoder."""
    # Convert attention mask to float to avoid dtype mismatch with scaled_dot_product_attention
    attention_mask = attention_mask.float()

    # Try different transformer backends
    transformer_backend = None
    if hasattr(news_encoder, "bert"):
        transformer_backend = news_encoder.bert
    elif hasattr(news_encoder, "roberta"):
        transformer_backend = news_encoder.roberta
    elif hasattr(news_encoder, "distilbert"):
        transformer_backend = news_encoder.distilbert
    elif hasattr(news_encoder, "transformer"):
        transformer_backend = news_encoder.transformer
    elif hasattr(news_encoder, "encoder"):
        transformer_backend = news_encoder.encoder

    if transformer_backend is not None:
        encoder_output = transformer_backend(
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
    elif hasattr(news_encoder, "lm"):
        encoder_output = news_encoder.lm(
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
    # Get the proper extended attention mask for transformer models
    # BERT expects attention_mask to be [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
    # for scaled_dot_product_attention to work properly with batched inputs

    # Ensure attention_mask is the correct dtype (float)
    if attention_mask.dtype not in [torch.float16, torch.float32, torch.float64]:
        attention_mask = attention_mask.float()

    # Try different transformer backends
    transformer_backend = None
    transformer_encoder = None

    if hasattr(news_encoder, "bert"):
        transformer_backend = news_encoder.bert
        transformer_encoder = news_encoder.bert.encoder
    elif hasattr(news_encoder, "roberta"):
        transformer_backend = news_encoder.roberta
        transformer_encoder = news_encoder.roberta.encoder
    elif hasattr(news_encoder, "distilbert"):
        transformer_backend = news_encoder.distilbert
        transformer_encoder = news_encoder.distilbert.transformer
    elif hasattr(news_encoder, "transformer") and hasattr(
        news_encoder.transformer, "encoder"
    ):
        transformer_backend = news_encoder.transformer
        transformer_encoder = news_encoder.transformer.encoder
    elif hasattr(news_encoder, "encoder") and hasattr(news_encoder.encoder, "encoder"):
        transformer_backend = news_encoder.encoder
        transformer_encoder = news_encoder.encoder.encoder

    if transformer_encoder is not None and transformer_backend is not None:
        # Create extended attention mask properly
        # Shape: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
        batch_size, seq_length = attention_mask.shape

        # Expand dimensions for broadcasting
        extended_attention_mask = attention_mask[
            :, None, None, :
        ]  # [batch_size, 1, 1, seq_len]

        # Get the dtype from embeddings to ensure compatibility
        extended_attention_mask = extended_attention_mask.to(dtype=embeddings.dtype)

        # Invert mask (1.0 for tokens to attend, 0.0 for masked tokens)
        # Then convert to additive mask (-inf for masked, 0 for unmasked)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
            embeddings.dtype
        ).min

        # Pass through transformer encoder
        encoder_output = transformer_encoder(
            embeddings, attention_mask=extended_attention_mask
        )

        # Extract sequence output from encoder output
        if hasattr(encoder_output, "last_hidden_state"):
            sequence_output = encoder_output.last_hidden_state
        elif isinstance(encoder_output, tuple):
            sequence_output = encoder_output[0]
        else:
            sequence_output = encoder_output

        # Apply final attention/pooling
        if hasattr(news_encoder, "additive_attention"):
            news_emb = news_encoder.additive_attention(sequence_output)
        else:
            news_emb = sequence_output[:, 0, :]  # CLS token
    elif hasattr(news_encoder, "lm"):
        # Get properly formatted attention mask for lm-based models
        batch_size, seq_length = attention_mask.shape

        # Expand dimensions for broadcasting
        extended_attention_mask = attention_mask[
            :, None, None, :
        ]  # [batch_size, 1, 1, seq_len]

        # Get the dtype from embeddings to ensure compatibility
        extended_attention_mask = extended_attention_mask.to(dtype=embeddings.dtype)

        # Invert mask (1.0 for tokens to attend, 0.0 for masked tokens)
        # Then convert to additive mask (-inf for masked, 0 for unmasked)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
            embeddings.dtype
        ).min

        # Pass through BERT encoder
        encoder_output = news_encoder.lm.encoder(
            embeddings, attention_mask=extended_attention_mask
        )

        # Extract sequence output from encoder output
        if hasattr(encoder_output, "last_hidden_state"):
            sequence_output = encoder_output.last_hidden_state
        elif isinstance(encoder_output, tuple):
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
    attributions_clean: Dict,
    attributions_poisoned: Dict,
    top_k: int = 20,
    process_body: bool = False,
) -> Dict:
    """
    Analyze which words are most important for real vs fake classification.

    Args:
        attributions_clean: Attributions from clean model
        attributions_poisoned: Attributions from poisoned model
        top_k: Number of top words to analyze
        process_body: If True and body attributions available (NAML), process body text

    Returns:
        Dictionary with analysis results (title and optionally body)
    """
    # Determine if we should process body
    has_body = (
        "body_attributions" in attributions_clean
        and "body_attributions" in attributions_poisoned
    )

    # Process title attributions (always)
    title_results = _process_attributions_for_importance(
        attributions_clean,
        attributions_poisoned,
        attr_key="attributions",
        token_key="tokens",
        top_k=top_k,
    )

    result = {"title": title_results}

    # Process body attributions if available and requested
    if has_body and process_body:
        body_results = _process_attributions_for_importance(
            attributions_clean,
            attributions_poisoned,
            attr_key="body_attributions",
            token_key="body_tokens",
            top_k=top_k,
        )
        result["body"] = body_results

    return result


def _process_attributions_for_importance(
    attributions_clean: Dict,
    attributions_poisoned: Dict,
    attr_key: str = "attributions",
    token_key: str = "tokens",
    top_k: int = 20,
) -> Dict:
    """
    Internal function to process attributions for importance analysis.

    Args:
        attributions_clean: Attributions from clean model
        attributions_poisoned: Attributions from poisoned model
        attr_key: Key for attributions (e.g., "attributions" or "body_attributions")
        token_key: Key for tokens (e.g., "tokens" or "body_tokens")
        top_k: Number of top words to analyze

    Returns:
        Dictionary with aggregated word importance
    """
    results = {
        "clean": {"real": defaultdict(list), "fake": defaultdict(list)},
        "poisoned": {"real": defaultdict(list), "fake": defaultdict(list)},
    }

    def process_model_attributions(attributions_dict, model_key):
        """Process attributions for one model."""
        for attr, words, label in zip(
            attributions_dict[attr_key],
            attributions_dict[token_key],
            attributions_dict["labels"],
        ):
            label_key = "fake" if label == 1 else "real"

            # Get top-k attributed words (top_k positive + top_k negative)
            if len(attr) > 0:
                # Get top_k positive attributions
                positive_mask = attr > 0
                top_positive_indices = np.array([], dtype=int)
                if np.any(positive_mask):
                    positive_indices = np.where(positive_mask)[0]
                    positive_scores = attr[positive_indices]
                    # Sort positive scores and get top_k highest
                    sorted_pos_idx = np.argsort(positive_scores)[-top_k:]
                    top_positive_indices = positive_indices[sorted_pos_idx]

                # Get top_k negative attributions
                negative_mask = attr < 0
                top_negative_indices = np.array([], dtype=int)
                if np.any(negative_mask):
                    negative_indices = np.where(negative_mask)[0]
                    negative_scores = attr[negative_indices]
                    # Sort negative scores and get top_k lowest (most negative)
                    sorted_neg_idx = np.argsort(negative_scores)[:top_k]
                    top_negative_indices = negative_indices[sorted_neg_idx]

                # Combine positive and negative indices
                top_indices = np.concatenate(
                    [top_positive_indices, top_negative_indices]
                )

                for idx in top_indices:
                    if idx < len(words):
                        word = words[idx]
                        score = attr[idx]

                        # Words are already cleaned (special tokens removed, subwords merged)
                        # by group_tokens_to_words(), so we can use them directly
                        results[model_key][label_key][word].append(float(score))

    process_model_attributions(attributions_clean, "clean")
    process_model_attributions(attributions_poisoned, "poisoned")

    # Aggregate scores
    aggregated = {
        "clean": {"real": {}, "fake": {}},
        "poisoned": {"real": {}, "fake": {}},
    }

    for model_key in ["clean", "poisoned"]:
        for label_key in ["real", "fake"]:
            for word, scores in results[model_key][label_key].items():
                aggregated[model_key][label_key][word] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "count": len(scores),
                }

    return aggregated


def analyze_word_frequency_from_top_samples(
    attributions_clean: Dict, attributions_poisoned: Dict, top_k_sample: int = 10
) -> Dict:
    """
    Analyze word frequency from top-k words per sample.

    This is an alternative approach to word importance analysis:
    1. For each sample, get top_k_sample positive words and top_k_sample negative words
    2. Combine all words from all samples
    3. Split words by space, filter stopwords
    4. Rank words by frequency separately for positive and negative attributions

    Args:
        attributions_clean: Attributions from clean model
        attributions_poisoned: Attributions from poisoned model
        top_k_sample: Number of top words to take from each sample (for pos and neg separately)

    Returns:
        Dictionary with word frequency analysis results separated by positive/negative
    """
    results = {
        "clean": {
            "real": {"positive": {}, "negative": {}},
            "fake": {"positive": {}, "negative": {}},
        },
        "poisoned": {
            "real": {"positive": {}, "negative": {}},
            "fake": {"positive": {}, "negative": {}},
        },
    }

    def process_model_attributions(attributions_dict, model_key):
        """Process attributions for one model."""
        # Collect word frequency from all samples - separate by positive/negative
        positive_word_frequency = {"real": defaultdict(int), "fake": defaultdict(int)}
        positive_word_attributions = {
            "real": defaultdict(list),
            "fake": defaultdict(list),
        }
        negative_word_frequency = {"real": defaultdict(int), "fake": defaultdict(int)}
        negative_word_attributions = {
            "real": defaultdict(list),
            "fake": defaultdict(list),
        }

        sample_count = {"real": 0, "fake": 0}

        for attr, words, label in zip(
            attributions_dict["attributions"],
            attributions_dict["tokens"],
            attributions_dict["labels"],
        ):
            label_key = "fake" if label == 1 else "real"
            sample_count[label_key] += 1

            if len(attr) == 0:
                continue

            # Get top_k_sample positive words from this sample
            positive_mask = attr > 0
            if np.any(positive_mask):
                positive_indices = np.where(positive_mask)[0]
                positive_scores = attr[positive_indices]
                # Sort and get top_k_sample highest positive scores
                sorted_pos_idx = np.argsort(positive_scores)[-top_k_sample:]
                top_positive_indices = positive_indices[sorted_pos_idx]

                # Process top positive words from this sample
                for idx in top_positive_indices:
                    if idx >= len(words):
                        continue

                    word = words[idx]
                    attribution = float(attr[idx])

                    # Split word by space to handle multi-word tokens
                    word_parts = word.split()

                    for word_part in word_parts:
                        # Convert to lowercase and filter stopwords
                        word_lower = word_part.lower().strip()

                        # Skip empty strings, stopwords, and very short words
                        if (
                            not word_lower
                            or word_lower in STOPWORDS
                            or len(word_lower) < 0
                        ):
                            continue

                        positive_word_frequency[label_key][word_lower] += 1
                        positive_word_attributions[label_key][word_lower].append(
                            attribution
                        )

            # Get top_k_sample negative words from this sample
            negative_mask = attr < 0
            if np.any(negative_mask):
                negative_indices = np.where(negative_mask)[0]
                negative_scores = attr[negative_indices]
                # Sort and get top_k_sample lowest (most negative) scores
                sorted_neg_idx = np.argsort(negative_scores)[:top_k_sample]
                top_negative_indices = negative_indices[sorted_neg_idx]

                # Process top negative words from this sample
                for idx in top_negative_indices:
                    if idx >= len(words):
                        continue

                    word = words[idx]
                    attribution = float(attr[idx])

                    # Split word by space to handle multi-word tokens
                    word_parts = word.split()

                    for word_part in word_parts:
                        # Convert to lowercase and filter stopwords
                        word_lower = word_part.lower().strip()

                        # Skip empty strings, stopwords, and very short words
                        if (
                            not word_lower
                            or word_lower in STOPWORDS
                            or len(word_lower) < 2
                        ):
                            continue

                        negative_word_frequency[label_key][word_lower] += 1
                        negative_word_attributions[label_key][word_lower].append(
                            attribution
                        )

        # Store results for each label
        for label_key in ["real", "fake"]:
            # Store results for positive words
            for word, freq in positive_word_frequency[label_key].items():
                results[model_key][label_key]["positive"][word] = {
                    "frequency": freq,
                    "mean_attribution": np.mean(
                        positive_word_attributions[label_key][word]
                    ),
                    "std_attribution": np.std(
                        positive_word_attributions[label_key][word]
                    ),
                    "sample_count": sample_count[label_key],
                }

            # Store results for negative words
            for word, freq in negative_word_frequency[label_key].items():
                results[model_key][label_key]["negative"][word] = {
                    "frequency": freq,
                    "mean_attribution": np.mean(
                        negative_word_attributions[label_key][word]
                    ),
                    "std_attribution": np.std(
                        negative_word_attributions[label_key][word]
                    ),
                    "sample_count": sample_count[label_key],
                }

    process_model_attributions(attributions_clean, "clean")
    process_model_attributions(attributions_poisoned, "poisoned")

    return results


def plot_word_importance(word_importance: Dict, save_path: str, top_k: int = 15):
    """
    Visualize word importance for real vs fake classification.

    Handles both NRMS (title only) and NAML (title + body) formats.

    Args:
        word_importance: Dictionary from analyze_word_importance
        save_path: Path to save the plot
        top_k: Number of top words to display
    """
    # Check if this is NAML format (has both "title" and "body" keys)
    # or NRMS format (has only "title" key or "clean"/"poisoned" keys directly)
    if "body" in word_importance:
        # NAML format - has both title and body, plot separately
        _plot_word_importance_naml(word_importance, save_path, top_k)
    elif "title" in word_importance:
        # NRMS format with new structure - has only "title" key
        _plot_word_importance_single(
            word_importance["title"], save_path, top_k, view_name="Title"
        )
    else:
        # NRMS format with old structure - has "clean"/"poisoned" keys directly
        _plot_word_importance_single(
            word_importance, save_path, top_k, view_name="Title"
        )


def _plot_word_importance_single(
    word_importance: Dict, save_path: str, top_k: int = 15, view_name: str = ""
):
    """Plot word importance for a single view (title or body)."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Plot for each model and label combination
    for i, model_key in enumerate(["clean", "poisoned"]):
        for j, label_key in enumerate(["real", "fake"]):
            ax = axes[i, j]

            # Check if model data exists
            if model_key not in word_importance:
                ax.text(
                    0.5,
                    0.5,
                    f"No {model_key} model data",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_title(f"{model_key.capitalize()} - {label_key.capitalize()}")
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                continue

            # Check if label data exists
            if label_key not in word_importance[model_key]:
                ax.text(
                    0.5,
                    0.5,
                    f"No {label_key} data",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_title(f"{model_key.capitalize()} - {label_key.capitalize()}")
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                continue

            # Get top words by mean attribution
            words_data = word_importance[model_key][label_key]
            if not words_data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
                ax.set_title(f"{model_key.capitalize()} - {label_key.capitalize()}")
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                continue

            # Get top_k positive words (highest positive mean attributions)
            positive_words = [
                (word, stats) for word, stats in words_data.items() if stats["mean"] > 0
            ]
            sorted_positive = sorted(
                positive_words, key=lambda x: x[1]["mean"], reverse=True
            )[:top_k]

            # Get top_k negative words (lowest/most negative mean attributions)
            negative_words = [
                (word, stats) for word, stats in words_data.items() if stats["mean"] < 0
            ]
            sorted_negative = sorted(negative_words, key=lambda x: x[1]["mean"])[:top_k]

            # Combine them (positive first, then negative)
            sorted_words = sorted_positive + sorted_negative

            if not sorted_words:
                ax.text(
                    0.5,
                    0.5,
                    "No significant words",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_title(f"{model_key.capitalize()} - {label_key.capitalize()}")
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
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
            title_suffix = f" ({view_name})" if view_name else ""
            ax.set_title(
                f"{model_key.capitalize()} Model - {label_key.capitalize()} News{title_suffix}",
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

    # Print diagnostic information
    print(f"Saved word importance plot to: {save_path}")
    for model_key in ["clean", "poisoned"]:
        if model_key not in word_importance:
            print(f"  Warning: No {model_key} model data found in word importance")
        else:
            for label_key in ["real", "fake"]:
                if label_key not in word_importance[model_key]:
                    print(f"  Warning: No {label_key} data found for {model_key} model")
                elif not word_importance[model_key][label_key]:
                    print(f"  Warning: Empty {label_key} data for {model_key} model")


def _plot_word_importance_naml(word_importance: Dict, save_path, top_k: int = 15):
    """Plot word importance for NAML with separate title and body views."""
    from pathlib import Path

    # Convert to Path if it's a string
    save_path = Path(save_path) if isinstance(save_path, str) else save_path

    # Plot title attributions
    if "title" in word_importance:
        title_path = save_path.parent / save_path.name.replace(".png", "_title.png")
        _plot_word_importance_single(
            word_importance["title"], str(title_path), top_k, view_name="Title"
        )

    # Plot body attributions if available
    if "body" in word_importance:
        body_path = save_path.parent / save_path.name.replace(".png", "_body.png")
        _plot_word_importance_single(
            word_importance["body"], str(body_path), top_k, view_name="Body"
        )
        print(f"Saved NAML word importance plots (title and body) to: {save_path}")


def plot_word_frequency_from_top_samples(
    word_frequency: Dict, save_path: str, top_k: int = 15
):
    """
    Visualize word frequency from top-k most affected samples.

    This plots words ranked by frequency (how often they appear in the most
    affected samples) as primary ranking, then by attribution score as secondary
    ranking. Words are separated by positive and negative attributions.

    Args:
        word_frequency: Dictionary from analyze_word_frequency_from_top_samples
        save_path: Path to save the plot
        top_k: Number of top words to display (top_k positive + top_k negative)
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    # Plot for each model and label combination
    for i, model_key in enumerate(["clean", "poisoned"]):
        for j, label_key in enumerate(["real", "fake"]):
            ax = axes[i, j]

            # Check if model data exists
            if model_key not in word_frequency:
                ax.text(
                    0.5,
                    0.5,
                    f"No {model_key} model data",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_title(f"{model_key.capitalize()} - {label_key.capitalize()}")
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                continue

            # Check if label data exists
            if label_key not in word_frequency[model_key]:
                ax.text(
                    0.5,
                    0.5,
                    f"No {label_key} data",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_title(f"{model_key.capitalize()} - {label_key.capitalize()}")
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                continue

            # Get positive and negative words data
            data = word_frequency[model_key][label_key]
            positive_data = data.get("positive", {})
            negative_data = data.get("negative", {})

            if not positive_data and not negative_data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
                ax.set_title(f"{model_key.capitalize()} - {label_key.capitalize()}")
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                continue

            # Get top_k positive words (sorted by frequency first, then attribution)
            sorted_positive = sorted(
                positive_data.items(),
                key=lambda x: (x[1]["frequency"], x[1]["mean_attribution"]),
                reverse=True,
            )[:top_k]

            # Get top_k negative words (sorted by frequency first, then attribution magnitude)
            sorted_negative = sorted(
                negative_data.items(),
                key=lambda x: (x[1]["frequency"], -x[1]["mean_attribution"]),
                reverse=True,
            )[:top_k]

            # Combine them (positive first, then negative)
            sorted_words = sorted_positive + sorted_negative

            if not sorted_words:
                ax.text(
                    0.5,
                    0.5,
                    "No significant words",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_title(f"{model_key.capitalize()} - {label_key.capitalize()}")
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                continue

            # Extract data
            words = [item[0] for item in sorted_words]
            frequencies = [item[1]["frequency"] for item in sorted_words]
            mean_attrs = [item[1]["mean_attribution"] for item in sorted_words]
            sample_count = sorted_words[0][1]["sample_count"] if sorted_words else 0

            # Create bar plot
            # Color by attribution (green for positive, red for negative)
            colors = ["green" if attr > 0 else "red" for attr in mean_attrs]

            bars = ax.barh(range(len(words)), frequencies, color=colors, alpha=0.7)

            # Add frequency labels on bars
            for idx, (bar, freq, attr) in enumerate(zip(bars, frequencies, mean_attrs)):
                width = bar.get_width()
                ax.text(
                    width + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    f"{freq} ({attr:+.3f})",
                    va="center",
                    fontsize=7,
                )

            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=8)
            ax.set_xlabel(f"Frequency (out of {sample_count} samples)", fontsize=10)
            ax.set_title(
                f"{model_key.capitalize()} Model - {label_key.capitalize()} News\n"
                f"Top-{top_k} Positive (green) + Top-{top_k} Negative (red) Words\n"
                f"Ranked by frequency, then attribution (from {sample_count} samples, stopwords removed)",
                fontsize=11,
                fontweight="bold",
            )
            ax.grid(axis="x", alpha=0.3)

            # Invert y-axis so most frequent is on top
            ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved word frequency plot to: {save_path}")


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

        words = attributions["tokens"][
            sample_idx
        ]  # Note: key is still "tokens" but contains words after grouping
        attr = attributions["attributions"][sample_idx]
        label = attributions["labels"][sample_idx]

        # Words are already cleaned (special tokens removed, subwords merged)
        # by group_tokens_to_words(), so we can use them directly

        # Limit to reasonable length
        max_len = 30
        if len(words) > max_len:
            words = words[:max_len]
            attr = attr[:max_len]

        # Create heatmap
        attr_normalized = np.array(attr).reshape(1, -1)
        im = ax.imshow(
            attr_normalized, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.1
        )

        # Set ticks
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha="right", fontsize=8)
        ax.set_yticks([])

        label_str = "Fake" if label == 1 else "Real"
        ax.set_ylabel(label_str, fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved attribution heatmap to: {save_path}")


def compare_attributions(
    clean_importance: Dict, poisoned_importance: Dict, save_path: str, top_k: int = 15
):
    """
    Compare word importance between clean and poisoned models.

    Args:
        clean_importance: Word importance from clean model
        poisoned_importance: Word importance from poisoned model
        save_path: Path to save comparison results
        top_k: Number of top positive and negative changes to display
    """
    # Handle new structure with "title" key (and optionally "body")
    # Use title importance for the comparison
    clean_title = clean_importance.get("title", clean_importance)
    poisoned_title = poisoned_importance.get("title", poisoned_importance)

    # Find words that changed importance significantly
    changes = {"real": {}, "fake": {}}

    for label in ["real", "fake"]:
        clean_words = set(clean_title["clean"][label].keys())
        poisoned_words = set(poisoned_title["poisoned"][label].keys())

        # Common words
        common = clean_words & poisoned_words

        for word in common:
            clean_score = clean_title["clean"][label][word]["mean"]
            poisoned_score = poisoned_title["poisoned"][label][word]["mean"]

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
            if poisoned_title["poisoned"][label][word]["count"] >= 3:
                changes[label][word] = {
                    "clean": 0.0,
                    "poisoned": poisoned_title["poisoned"][label][word]["mean"],
                    "change": poisoned_title["poisoned"][label][word]["mean"],
                }

    # Visualize changes
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for i, label in enumerate(["real", "fake"]):
        ax = axes[i]

        if not changes[label]:
            ax.text(0.5, 0.5, "No significant changes", ha="center", va="center")
            ax.set_title(f"{label.capitalize()} News")
            continue

        # Get top_k positive changes (largest positive changes)
        positive_changes = [
            (word, data) for word, data in changes[label].items() if data["change"] > 0
        ]
        sorted_positive = sorted(
            positive_changes, key=lambda x: x[1]["change"], reverse=True
        )[:top_k]

        # Get top_k negative changes (largest negative changes)
        negative_changes = [
            (word, data) for word, data in changes[label].items() if data["change"] < 0
        ]
        sorted_negative = sorted(negative_changes, key=lambda x: x[1]["change"])[:top_k]

        # Combine them
        sorted_changes = sorted_positive + sorted_negative

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
