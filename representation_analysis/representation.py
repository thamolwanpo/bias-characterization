"""
Extract news encoder representations from model.
"""

import torch
import numpy as np

import sys
import os

sys.path.insert(0, os.path.abspath("../plm4newsrs/src/models"))
from registry import get_model_class


def extract_news_representations(data_loader, config, model_config):
    """
    Extract news encoder representations with metadata.

    Args:
        data_loader: DataLoader providing data batches (expects BenchmarkDataset with impression_data)
        config: configuration dict with keys: 'device', 'model_checkpoint'
        model_config: model configuration with key: 'architecture'

    Returns:
        dict with:
            - embeddings: torch.Tensor of shape [n_samples, n_candidates, embedding_dim]
            - user_embeddings: torch.Tensor of shape [n_samples, embedding_dim]
            - interactions: torch.Tensor of shape [n_samples, n_candidates, embedding_dim] - element-wise product of user and news embeddings
            - scores: torch.Tensor of shape [n_samples, n_candidates] - raw dot product scores (user_emb @ news_emb.T)
            - prediction_scores: torch.Tensor of shape [n_samples, n_candidates] - model's click probabilities
            - prediction_ranks: torch.Tensor of shape [n_samples, n_candidates] - rankings (1=highest score)
            - is_fake: np.ndarray of shape [n_samples, n_candidates] - boolean fake/real labels
            - titles: np.ndarray of shape [n_samples, n_candidates] - news article titles
            - candidate_ids: np.ndarray of shape [n_samples, n_candidates] - news article IDs
            - user_ids: np.ndarray of shape [n_samples] - user IDs
    """

    # Load model
    lit_model = load_model(config, model_config)
    lit_model.eval()

    device = config.get("device", "cpu")
    lit_model = lit_model.to(device)

    # Get the underlying recommender model and news encoder
    recommender_model = lit_model.model
    news_encoder = recommender_model.news_encoder
    user_encoder = recommender_model.user_encoder

    # Extract representations
    all_embeddings = []
    all_user_embeddings = []
    all_prediction_scores = []
    all_prediction_ranks = []
    all_interactions = []  # Element-wise product: user_emb * news_emb
    all_scores = []  # Raw scores from model forward pass
    all_is_fake = []
    all_titles = []
    all_candidate_ids = []
    all_user_ids = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch is None:
                continue

            # Move batch to device
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }

            # Get full model predictions (scores for ranking)
            # This will give us click probability scores for each candidate
            scores = lit_model.model(batch)  # Shape: (batch_size, num_candidates)

            # Calculate rankings (1 = highest predicted probability, N = lowest)
            # argsort gives indices from low to high, so we reverse it
            ranks = (
                torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1) + 1
            )

            # Store prediction scores and ranks
            all_prediction_scores.append(scores.cpu())
            all_prediction_ranks.append(ranks.cpu())

            # Extract user IDs from batch
            if "user_id" in batch:
                all_user_ids.extend(batch["user_id"])

            # Extract is_fake labels from impression_data (BenchmarkDataset format)
            if "impression_data" in batch:
                # impression_data[batch_idx][candidate_idx] = (candidate_id, title, label, is_fake)
                batch_is_fake = []
                batch_titles = []
                batch_ids = []

                for impression in batch["impression_data"]:
                    impression_is_fake = []
                    impression_titles = []
                    impression_ids = []

                    for candidate_id, title, label, is_fake in impression:
                        impression_is_fake.append(is_fake)
                        impression_titles.append(title)
                        impression_ids.append(candidate_id)

                    batch_is_fake.append(impression_is_fake)
                    batch_titles.append(impression_titles)
                    batch_ids.append(impression_ids)

                all_is_fake.append(batch_is_fake)
                all_titles.append(batch_titles)
                all_candidate_ids.append(batch_ids)

            # Extract news representations based on model architecture
            architecture = model_config.architecture

            if architecture in ["simple", "nrms"]:
                # Simple and NRMS models process news similarly
                if "candidate_input_ids" in batch:
                    # Transformer mode
                    batch_size, num_cands, seq_len = batch["candidate_input_ids"].shape
                    cand_ids = batch["candidate_input_ids"].view(-1, seq_len)
                    cand_mask = batch["candidate_attention_mask"].view(-1, seq_len)

                    embeddings = news_encoder(
                        input_ids=cand_ids, attention_mask=cand_mask
                    )
                    # Reshape back to (batch_size, num_cands, embedding_dim)
                    embeddings = embeddings.view(batch_size, num_cands, -1)
                elif "candidate_titles" in batch:
                    # GloVe mode
                    candidate_titles = batch["candidate_titles"]
                    device_indicator = batch.get(
                        "device_indicator", torch.tensor([0], device=device)
                    )

                    # Process each item in batch
                    batch_embeddings = []
                    for candidate_list in candidate_titles:
                        embs = news_encoder(
                            input_ids=device_indicator, text_list=candidate_list
                        )
                        batch_embeddings.append(embs)

                    # Stack to maintain (batch_size, num_cands, embedding_dim)
                    embeddings = torch.stack(batch_embeddings, dim=0)
                else:
                    raise ValueError(
                        "Batch must contain either 'candidate_input_ids' or 'candidate_titles'"
                    )

            elif architecture == "naml":
                # NAML model with multi-view processing
                if "candidate_input_ids" in batch:
                    # Transformer mode
                    batch_size, num_cands, seq_len = batch["candidate_input_ids"].shape
                    cand_ids = batch["candidate_input_ids"].view(-1, seq_len)
                    cand_mask = batch["candidate_attention_mask"].view(-1, seq_len)

                    # Prepare optional inputs
                    body_ids = None
                    body_mask = None
                    if "candidate_body_input_ids" in batch:
                        body_ids = batch["candidate_body_input_ids"].view(-1, seq_len)
                        body_mask = batch["candidate_body_attention_mask"].view(
                            -1, seq_len
                        )

                    category_ids = None
                    if "candidate_categories" in batch:
                        category_ids = batch["candidate_categories"].view(-1)

                    subcategory_ids = None
                    if "candidate_subcategories" in batch:
                        subcategory_ids = batch["candidate_subcategories"].view(-1)

                    embeddings = news_encoder(
                        title_input_ids=cand_ids,
                        title_attention_mask=cand_mask,
                        body_input_ids=body_ids,
                        body_attention_mask=body_mask,
                        category_ids=category_ids,
                        subcategory_ids=subcategory_ids,
                    )
                    # Reshape back to (batch_size, num_cands, embedding_dim)
                    embeddings = embeddings.view(batch_size, num_cands, -1)
                elif "candidate_titles" in batch:
                    # GloVe mode
                    candidate_titles = batch["candidate_titles"]
                    device_indicator = batch.get(
                        "device_indicator", torch.tensor([0], device=device)
                    )

                    # Optional body text
                    candidate_bodies = batch.get("candidate_bodies", None)

                    # Process each item in batch
                    batch_embeddings = []
                    for idx, candidate_list in enumerate(candidate_titles):
                        body_list = candidate_bodies[idx] if candidate_bodies else None

                        embs = news_encoder(
                            title_text_list=candidate_list,
                            body_text_list=body_list,
                            device_indicator=device_indicator,
                        )
                        batch_embeddings.append(embs)

                    # Stack to maintain (batch_size, num_cands, embedding_dim)
                    embeddings = torch.stack(batch_embeddings, dim=0)
                else:
                    raise ValueError(
                        "Batch must contain either 'candidate_input_ids' or 'candidate_titles'"
                    )
            else:
                raise ValueError(f"Unknown architecture: {architecture}")

            # Collect embeddings
            all_embeddings.append(embeddings.cpu())

            # Extract user embeddings
            # User encoder typically processes history news
            if "history_input_ids" in batch:
                # Transformer mode
                batch_size, hist_len, seq_len = batch["history_input_ids"].shape
                hist_ids = batch["history_input_ids"].view(-1, seq_len)
                hist_mask = batch["history_attention_mask"].view(-1, seq_len)

                # Get history news embeddings based on architecture
                if architecture == "naml":
                    # NAML uses different parameter names
                    # Prepare optional inputs for history
                    hist_body_ids = None
                    hist_body_mask = None
                    if "history_body_input_ids" in batch:
                        hist_body_ids = batch["history_body_input_ids"].view(-1, seq_len)
                        hist_body_mask = batch["history_body_attention_mask"].view(-1, seq_len)

                    hist_category_ids = None
                    if "history_categories" in batch:
                        hist_category_ids = batch["history_categories"].view(-1)

                    hist_subcategory_ids = None
                    if "history_subcategories" in batch:
                        hist_subcategory_ids = batch["history_subcategories"].view(-1)

                    hist_news_embeddings = news_encoder(
                        title_input_ids=hist_ids,
                        title_attention_mask=hist_mask,
                        body_input_ids=hist_body_ids,
                        body_attention_mask=hist_body_mask,
                        category_ids=hist_category_ids,
                        subcategory_ids=hist_subcategory_ids,
                    )
                else:
                    # Simple and NRMS architectures
                    hist_news_embeddings = news_encoder(
                        input_ids=hist_ids, attention_mask=hist_mask
                    )

                hist_news_embeddings = hist_news_embeddings.view(batch_size, hist_len, -1)

                # Get user embeddings
                user_embeddings = user_encoder(hist_news_embeddings)
            elif "history_titles" in batch:
                # GloVe mode
                history_titles = batch["history_titles"]
                device_indicator = batch.get(
                    "device_indicator", torch.tensor([0], device=device)
                )

                # Optional body text
                history_bodies = batch.get("history_bodies", None)

                # Process each user's history
                batch_user_embeddings = []
                for idx, history_list in enumerate(history_titles):
                    body_list = history_bodies[idx] if history_bodies else None

                    if architecture == "naml":
                        # NAML uses different parameter names
                        hist_embs = news_encoder(
                            title_text_list=history_list,
                            body_text_list=body_list,
                            device_indicator=device_indicator,
                        )
                    else:
                        # Simple and NRMS architectures
                        hist_embs = news_encoder(
                            input_ids=device_indicator, text_list=history_list
                        )

                    # hist_embs shape: (hist_len, embedding_dim)
                    # Add batch dimension for user_encoder
                    hist_embs = hist_embs.unsqueeze(0)  # (1, hist_len, embedding_dim)
                    user_emb = user_encoder(hist_embs)
                    batch_user_embeddings.append(user_emb.squeeze(0))

                user_embeddings = torch.stack(batch_user_embeddings, dim=0)
            else:
                raise ValueError(
                    "Batch must contain either 'history_input_ids' or 'history_titles'"
                )

            all_user_embeddings.append(user_embeddings.cpu())

            # Compute element-wise interaction: user_emb * news_emb
            # user_embeddings shape: (batch_size, embedding_dim)
            # embeddings shape: (batch_size, num_candidates, embedding_dim)
            # Expand user_embeddings to match: (batch_size, 1, embedding_dim)
            user_emb_expanded = user_embeddings.unsqueeze(1)  # (batch_size, 1, embedding_dim)
            interaction = user_emb_expanded * embeddings  # (batch_size, num_candidates, embedding_dim)
            all_interactions.append(interaction.cpu())

            # Compute scores: batch matrix multiplication
            # This is the same computation as in the model forward pass
            # scores = candidate_embeddings @ user_embedding.T
            batch_scores = torch.bmm(
                embeddings,  # (batch_size, num_candidates, embedding_dim)
                user_embeddings.unsqueeze(-1),  # (batch_size, embedding_dim, 1)
            ).squeeze(dim=-1)  # (batch_size, num_candidates)
            all_scores.append(batch_scores.cpu())

    # Concatenate all batches
    embeddings = torch.cat(all_embeddings, dim=0)
    user_embeddings = torch.cat(all_user_embeddings, dim=0)
    prediction_scores = torch.cat(all_prediction_scores, dim=0)
    prediction_ranks = torch.cat(all_prediction_ranks, dim=0)
    interactions = torch.cat(all_interactions, dim=0)
    scores = torch.cat(all_scores, dim=0)

    # Flatten nested lists for is_fake, titles, and candidate_ids
    # From list of batches of impressions to flat lists matching shape
    is_fake_flat = []
    titles_flat = []
    candidate_ids_flat = []

    for batch_is_fake, batch_titles, batch_ids in zip(all_is_fake, all_titles, all_candidate_ids):
        is_fake_flat.extend(batch_is_fake)
        titles_flat.extend(batch_titles)
        candidate_ids_flat.extend(batch_ids)

    # Convert to numpy arrays with shape (n_samples, n_candidates)
    is_fake_array = np.array(is_fake_flat) if is_fake_flat else None
    titles_array = np.array(titles_flat) if titles_flat else None
    candidate_ids_array = np.array(candidate_ids_flat) if candidate_ids_flat else None
    user_ids_array = np.array(all_user_ids) if all_user_ids else None

    # Check for user-side poisoning by comparing user-news interaction scores
    if is_fake_array is not None:
        # Reshape embeddings to (n_samples * n_candidates, embedding_dim) for filtering
        embedding_dim = embeddings.shape[-1]
        news_emb_flat = embeddings.view(-1, embedding_dim)
        is_fake_flat_bool = is_fake_array.flatten()

        # Separate real and fake news embeddings
        real_news_mask = ~is_fake_flat_bool
        fake_news_mask = is_fake_flat_bool

        real_news_emb = news_emb_flat[real_news_mask]
        fake_news_emb = news_emb_flat[fake_news_mask]

        # Calculate average scores: user_emb @ news_emb.T
        # For each user, compute avg similarity with real/fake news
        if len(real_news_emb) > 0 and len(fake_news_emb) > 0:
            # Compute similarity scores: (n_samples, n_real_news) and (n_samples, n_fake_news)
            scores_real = user_embeddings @ real_news_emb.T  # (n_samples, n_real_news)
            scores_fake = user_embeddings @ fake_news_emb.T  # (n_samples, n_fake_news)

            # Average scores across all real/fake news
            avg_score_real = scores_real.mean().item()
            avg_score_fake = scores_fake.mean().item()

            print(f"\n=== User-News Interaction Analysis ===")
            print(f"Average user-real_news score: {avg_score_real:.4f}")
            print(f"Average user-fake_news score: {avg_score_fake:.4f}")
            print(f"Difference (fake - real): {avg_score_fake - avg_score_real:.4f}")
            if avg_score_fake > avg_score_real:
                print("WARNING: User embeddings show higher affinity to fake news!")
                print("This suggests potential user-side poisoning.")
            print("=" * 40)

    return {
        "embeddings": embeddings,                # Shape: (n_samples, n_candidates, embedding_dim)
        "user_embeddings": user_embeddings,      # Shape: (n_samples, embedding_dim)
        "interactions": interactions,            # Shape: (n_samples, n_candidates, embedding_dim) - element-wise user * news
        "scores": scores,                        # Shape: (n_samples, n_candidates) - raw dot product scores
        "prediction_scores": prediction_scores,  # Shape: (n_samples, n_candidates) - click probability scores
        "prediction_ranks": prediction_ranks,    # Shape: (n_samples, n_candidates) - ranking (1=best, N=worst)
        "is_fake": is_fake_array,               # Shape: (n_samples, n_candidates) - boolean array
        "titles": titles_array,                  # Shape: (n_samples, n_candidates) - string array
        "candidate_ids": candidate_ids_array,    # Shape: (n_samples, n_candidates) - ID array
        "user_ids": user_ids_array,              # Shape: (n_samples,) - user ID array
    }


def load_model(config, model_config):
    """
    Load model from checkpoint.
    """
    model = get_model_class(model_config.architecture).load_from_checkpoint(
        config["model_checkpoint"], config=model_config
    )
    return model


def get_representation_info(embeddings):
    """
    Print information about extracted representations.
    """
    print(f"\nRepresentation Info:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")
    print(f"  Device: {embeddings.device}")
    print(f"  Mean: {embeddings.mean().item():.4f}")
    print(f"  Std: {embeddings.std().item():.4f}")
    print(f"  Min: {embeddings.min().item():.4f}")
    print(f"  Max: {embeddings.max().item():.4f}")
