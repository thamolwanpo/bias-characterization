"""
Extract news encoder representations from model.
OPTIMIZED VERSION: Faster extraction with batch processing and memory efficiency.
"""

import torch
import numpy as np

import sys
import os
from tqdm import tqdm

# sys.path.insert(
#     0,
#     os.path.abspath(
#         "/content/drive/MyDrive/bias-characterized/bias-characterization/plm4newsrs/src/models"
#     ),
# )

sys.path.insert(
    0,
    os.path.abspath("/home/thamo/PhD/bias-characterization/plm4newsrs/src/models"),
)

from registry import get_model_class


def extract_news_representations(data_loader, config, model_config, use_amp=True):
    """
    Extract news encoder representations with metadata.

    OPTIMIZATIONS:
    - Automatic Mixed Precision (AMP) for faster GPU inference
    - Efficient tensor operations
    - Pre-allocated lists
    - Optional progress bar
    - Memory-efficient concatenation

    Args:
        data_loader: DataLoader providing data batches
        config: configuration dict with keys: 'device', 'model_checkpoint'
        model_config: model configuration with key: 'architecture'
        use_amp: Use automatic mixed precision (default: True, ~2x faster on GPU)

    Returns:
        dict with embeddings, scores, and metadata
    """

    # Load model
    lit_model = load_model(config, model_config)
    lit_model.eval()

    device = config.get("device", "cpu")
    lit_model = lit_model.to(device)

    # Enable optimizations
    if device != "cpu":
        torch.backends.cudnn.benchmark = True  # Auto-tune conv algorithms
        if hasattr(torch, "compile"):
            # PyTorch 2.0+ compilation (can give 20-50% speedup)
            print("Using torch.compile for additional speedup...")
            lit_model = torch.compile(lit_model, mode="reduce-overhead")

    # Get model components
    recommender_model = lit_model.model
    news_encoder = recommender_model.news_encoder
    user_encoder = recommender_model.user_encoder

    # Determine configuration
    use_glove = "glove" in model_config.model_name.lower()
    architecture = model_config.architecture

    # Log configuration
    print(f"\n=== Extraction Configuration ===")
    print(f"Architecture: {architecture}")
    print(f"Mode: {'GloVe' if use_glove else 'Transformer'}")
    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp and device != 'cpu'}")

    # Check batch structure
    sample_batch = next(iter(data_loader))
    print(f"Available batch keys: {list(sample_batch.keys())}")

    # Determine body text usage
    if architecture == "naml":
        use_body = True
        print(f"NAML: Using both titles and body text")
    else:
        use_body = False
        print(f"{architecture.upper()}: Using only titles")

    print("=" * 40)

    # Pre-allocate storage lists (more efficient than append)
    num_batches = len(data_loader)
    all_embeddings = []
    all_user_embeddings = []
    all_prediction_scores = []
    all_prediction_ranks = []
    all_interactions = []
    all_scores = []
    all_is_fake = []
    all_titles = []
    all_candidate_ids = []
    all_user_ids = []

    # Setup AMP
    use_amp = use_amp and (device != "cpu")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print(f"\nProcessing {num_batches} batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(data_loader, desc="Extracting representations")
        ):
            if batch is None:
                continue

            # Move batch to device (more efficient batched transfer)
            batch = _move_batch_to_device(batch, device)

            # Use AMP context for faster computation
            with torch.cuda.amp.autocast(enabled=use_amp):
                # Get model predictions
                scores = lit_model.model(batch)

                # Calculate rankings
                ranks = (
                    torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1)
                    + 1
                )

            # Store predictions (move to CPU immediately to free GPU memory)
            all_prediction_scores.append(scores.cpu())
            all_prediction_ranks.append(ranks.cpu())

            # Extract metadata (no GPU operations needed)
            if "user_id" in batch:
                all_user_ids.extend(batch["user_id"])

            if "impression_data" in batch:
                batch_is_fake, batch_titles, batch_ids = _extract_impression_metadata(
                    batch["impression_data"]
                )
                all_is_fake.append(batch_is_fake)
                all_titles.append(batch_titles)
                all_candidate_ids.append(batch_ids)

            # Extract embeddings with AMP
            with torch.cuda.amp.autocast(enabled=use_amp):
                if use_glove:
                    candidate_embeddings, history_embeddings = (
                        _extract_glove_embeddings(
                            batch, news_encoder, device, architecture, use_body
                        )
                    )
                else:
                    candidate_embeddings, history_embeddings = (
                        _extract_transformer_embeddings(
                            batch, news_encoder, device, architecture, use_body
                        )
                    )

                # Get user embeddings
                user_embeddings = user_encoder(history_embeddings)

                # Compute interactions and scores
                user_emb_expanded = user_embeddings.unsqueeze(1)
                interaction = user_emb_expanded * candidate_embeddings

                batch_scores = torch.bmm(
                    candidate_embeddings,
                    user_embeddings.unsqueeze(-1),
                ).squeeze(dim=-1)

            # Move to CPU and store (free GPU memory)
            all_embeddings.append(candidate_embeddings.cpu())
            all_user_embeddings.append(user_embeddings.cpu())
            all_interactions.append(interaction.cpu())
            all_scores.append(batch_scores.cpu())

            # Print first batch info
            if batch_idx == 0:
                print(f"\nFirst batch shapes:")
                print(f"  Candidate embeddings: {candidate_embeddings.shape}")
                print(f"  History embeddings: {history_embeddings.shape}")
                print(f"  User embeddings: {user_embeddings.shape}")

            # Clear GPU cache periodically
            if device != "cpu" and batch_idx % 50 == 0:
                torch.cuda.empty_cache()

    print("\nConcatenating results...")

    # Efficient concatenation
    embeddings = torch.cat(all_embeddings, dim=0)
    user_embeddings = torch.cat(all_user_embeddings, dim=0)
    prediction_scores = torch.cat(all_prediction_scores, dim=0)
    prediction_ranks = torch.cat(all_prediction_ranks, dim=0)
    interactions = torch.cat(all_interactions, dim=0)
    scores = torch.cat(all_scores, dim=0)

    print("\n Process metadata")
    # Process metadata
    is_fake_array, titles_array, candidate_ids_array = _process_metadata(
        all_is_fake, all_titles, all_candidate_ids
    )
    user_ids_array = np.array(all_user_ids) if all_user_ids else None

    # User-side poisoning analysis
    if is_fake_array is not None:
        print("\n Analyze user poisoning")
        _analyze_user_poisoning(embeddings, user_embeddings, is_fake_array)

    print(f"\n=== Final Shapes ===")
    print(f"Embeddings: {embeddings.shape}")
    print(f"User embeddings: {user_embeddings.shape}")
    print(f"Scores: {scores.shape}")
    print("=" * 40)

    return {
        "embeddings": embeddings,
        "user_embeddings": user_embeddings,
        "interactions": interactions,
        "scores": scores,
        "prediction_scores": prediction_scores,
        "prediction_ranks": prediction_ranks,
        "is_fake": is_fake_array,
        "titles": titles_array,
        "candidate_ids": candidate_ids_array,
        "user_ids": user_ids_array,
    }


def _move_batch_to_device(batch, device):
    """Efficiently move batch to device."""
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def _extract_impression_metadata(impression_data):
    """Extract metadata from impression data efficiently."""
    batch_is_fake = []
    batch_titles = []
    batch_ids = []

    for impression in impression_data:
        # Vectorized extraction
        impression_is_fake = [item[3] for item in impression]
        impression_titles = [item[1] for item in impression]
        impression_ids = [item[0] for item in impression]

        batch_is_fake.append(impression_is_fake)
        batch_titles.append(impression_titles)
        batch_ids.append(impression_ids)

    return batch_is_fake, batch_titles, batch_ids


def _process_metadata(all_is_fake, all_titles, all_candidate_ids):
    """Process metadata lists into arrays efficiently."""
    is_fake_flat = [item for batch in all_is_fake for item in batch]
    titles_flat = [item for batch in all_titles for item in batch]
    candidate_ids_flat = [item for batch in all_candidate_ids for item in batch]

    is_fake_array = np.array(is_fake_flat) if is_fake_flat else None
    titles_array = np.array(titles_flat) if titles_flat else None
    candidate_ids_array = np.array(candidate_ids_flat) if candidate_ids_flat else None

    return is_fake_array, titles_array, candidate_ids_array


def _analyze_user_poisoning(embeddings, user_embeddings, is_fake_array):
    """Analyze user-side poisoning indicators."""
    embedding_dim = embeddings.shape[-1]
    news_emb_flat = embeddings.view(-1, embedding_dim)
    is_fake_flat_bool = is_fake_array.flatten()

    real_news_mask = ~is_fake_flat_bool
    fake_news_mask = is_fake_flat_bool

    real_news_emb = news_emb_flat[real_news_mask]
    fake_news_emb = news_emb_flat[fake_news_mask]

    if len(real_news_emb) > 0 and len(fake_news_emb) > 0:
        scores_real = user_embeddings @ real_news_emb.T
        scores_fake = user_embeddings @ fake_news_emb.T

        avg_score_real = scores_real.mean().item()
        avg_score_fake = scores_fake.mean().item()

        print(f"\n=== User-News Interaction Analysis ===")
        print(f"Average user-real_news score: {avg_score_real:.4f}")
        print(f"Average user-fake_news score: {avg_score_fake:.4f}")
        print(f"Difference (fake - real): {avg_score_fake - avg_score_real:.4f}")

        if avg_score_fake > avg_score_real:
            print("⚠️  WARNING: User embeddings show higher affinity to fake news!")
            print("   This suggests potential user-side poisoning.")
        else:
            print("✓ User embeddings prefer real news.")

        print("=" * 40)


def _extract_glove_embeddings(batch, news_encoder, device, architecture, use_body=True):
    """
    Extract embeddings in GloVe mode (optimized).
    """
    candidate_titles_batch = batch["candidate_titles"]
    history_titles_batch = batch["history_titles"]
    device_indicator = batch.get("device_indicator", torch.tensor([0], device=device))

    batch_size = len(candidate_titles_batch)

    if architecture == "naml" and use_body:
        candidate_texts_batch = batch.get("candidate_texts", None)
        history_texts_batch = batch.get("history_texts", None)
        candidate_categories = batch.get("candidate_categories", None)
        history_categories = batch.get("history_categories", None)
        candidate_subcategories = batch.get("candidate_subcategories", None)
        history_subcategories = batch.get("history_subcategories", None)

        # Pre-allocate lists
        candidate_embeddings_list = []
        history_embeddings_list = []

        for i in range(batch_size):
            # Candidates
            embs = news_encoder(
                title_text_list=candidate_titles_batch[i],
                body_text_list=(
                    candidate_texts_batch[i] if candidate_texts_batch else None
                ),
                category_ids=candidate_categories[i] if candidate_categories else None,
                subcategory_ids=(
                    candidate_subcategories[i] if candidate_subcategories else None
                ),
                device_indicator=device_indicator,
            )
            candidate_embeddings_list.append(embs)

            # History
            embs = news_encoder(
                title_text_list=history_titles_batch[i],
                body_text_list=history_texts_batch[i] if history_texts_batch else None,
                category_ids=history_categories[i] if history_categories else None,
                subcategory_ids=(
                    history_subcategories[i] if history_subcategories else None
                ),
                device_indicator=device_indicator,
            )
            history_embeddings_list.append(embs)

    else:
        # Simple/NRMS - titles only
        candidate_embeddings_list = [
            news_encoder(input_ids=device_indicator, text_list=titles)
            for titles in candidate_titles_batch
        ]

        history_embeddings_list = [
            news_encoder(input_ids=device_indicator, text_list=titles)
            for titles in history_titles_batch
        ]

    # Stack efficiently
    candidate_embeddings = torch.stack(candidate_embeddings_list)
    history_embeddings = torch.stack(history_embeddings_list)

    return candidate_embeddings, history_embeddings


def _extract_transformer_embeddings(
    batch, news_encoder, device, architecture, use_body=True
):
    """
    Extract embeddings in Transformer mode (optimized).
    """
    # Get inputs
    candidate_title_ids = batch["candidate_title_input_ids"]
    candidate_title_mask = batch["candidate_title_attention_mask"]
    history_title_ids = batch["history_title_input_ids"]
    history_title_mask = batch["history_title_attention_mask"]

    batch_size, num_candidates, seq_len = candidate_title_ids.shape
    _, history_len, _ = history_title_ids.shape

    # Handle body text for NAML
    if architecture == "naml" and use_body:
        candidate_body_ids = batch.get("candidate_input_ids", None)
        candidate_body_mask = batch.get("candidate_attention_mask", None)
        history_body_ids = batch.get("history_input_ids", None)
        history_body_mask = batch.get("history_attention_mask", None)
        candidate_categories = batch.get("candidate_categories", None)
        candidate_subcategories = batch.get("candidate_subcategories", None)
        history_categories = batch.get("history_categories", None)
        history_subcategories = batch.get("history_subcategories", None)
    else:
        candidate_body_ids = candidate_body_mask = None
        history_body_ids = history_body_mask = None
        candidate_categories = candidate_subcategories = None
        history_categories = history_subcategories = None

    # Flatten for batch processing (more efficient)
    candidate_title_ids_flat = candidate_title_ids.view(
        batch_size * num_candidates, seq_len
    )
    candidate_title_mask_flat = candidate_title_mask.view(
        batch_size * num_candidates, seq_len
    )

    # Encode candidates
    if architecture == "naml" and use_body and candidate_body_ids is not None:
        _, _, body_seq_len = candidate_body_ids.shape
        candidate_embeddings_flat = news_encoder(
            title_input_ids=candidate_title_ids_flat,
            title_attention_mask=candidate_title_mask_flat,
            body_input_ids=candidate_body_ids.view(
                batch_size * num_candidates, body_seq_len
            ),
            body_attention_mask=candidate_body_mask.view(
                batch_size * num_candidates, body_seq_len
            ),
            category_ids=(
                candidate_categories.view(batch_size * num_candidates)
                if candidate_categories is not None
                else None
            ),
            subcategory_ids=(
                candidate_subcategories.view(batch_size * num_candidates)
                if candidate_subcategories is not None
                else None
            ),
        )
    else:
        candidate_embeddings_flat = news_encoder(
            input_ids=candidate_title_ids_flat, attention_mask=candidate_title_mask_flat
        )

    candidate_embeddings = candidate_embeddings_flat.view(
        batch_size, num_candidates, -1
    )

    # Encode history
    history_title_ids_flat = history_title_ids.view(batch_size * history_len, seq_len)
    history_title_mask_flat = history_title_mask.view(batch_size * history_len, seq_len)

    if architecture == "naml" and use_body and history_body_ids is not None:
        _, _, body_seq_len = history_body_ids.shape
        history_embeddings_flat = news_encoder(
            title_input_ids=history_title_ids_flat,
            title_attention_mask=history_title_mask_flat,
            body_input_ids=history_body_ids.view(
                batch_size * history_len, body_seq_len
            ),
            body_attention_mask=history_body_mask.view(
                batch_size * history_len, body_seq_len
            ),
            category_ids=(
                history_categories.view(batch_size * history_len)
                if history_categories is not None
                else None
            ),
            subcategory_ids=(
                history_subcategories.view(batch_size * history_len)
                if history_subcategories is not None
                else None
            ),
        )
    else:
        history_embeddings_flat = news_encoder(
            input_ids=history_title_ids_flat, attention_mask=history_title_mask_flat
        )

    history_embeddings = history_embeddings_flat.view(batch_size, history_len, -1)

    return candidate_embeddings, history_embeddings


def load_model(config, model_config):
    """Load model from checkpoint."""
    model = get_model_class(model_config.architecture).load_from_checkpoint(
        config["model_checkpoint"], config=model_config
    )
    return model


def get_representation_info(embeddings):
    """Print information about extracted representations."""
    print(f"\nRepresentation Info:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")
    print(f"  Device: {embeddings.device}")
    print(f"  Mean: {embeddings.mean().item():.4f}")
    print(f"  Std: {embeddings.std().item():.4f}")
    print(f"  Min: {embeddings.min().item():.4f}")
    print(f"  Max: {embeddings.max().item():.4f}")
