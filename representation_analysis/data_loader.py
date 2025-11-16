"""
Data loader for test set with fake/real labels.
Handles sampling and filtering.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

import sys
import os
import ast

from torch.utils.data import Dataset
from tqdm import trange, tqdm

# sys.path.insert(
#     0,
#     os.path.abspath(
#         "/content/drive/MyDrive/bias-characterized/bias-characterization/plm4newsrs/src/evaluation"
#     ),
# )

sys.path.insert(
    0,
    os.path.abspath(
        "/workspace/plm4newsrs/src/evaluation"
    ),
)

from benchmark_dataset import benchmark_collate_fn, convert_pairwise_to_listwise_eval
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from typing import Optional, Dict, Any
import json


class BenchmarkDataset(Dataset):
    """Dataset for benchmark evaluation."""

    def __init__(
        self,
        csv_path,
        news_items_path,
        tokenizer,
        config,
        user_map: Optional[Dict[Any, int]] = None,
    ):
        """
        Args:
            csv_path: Path to benchmark CSV file
            tokenizer: HuggingFace tokenizer (None for GloVe)
            config: Base configuration
        """
        print(f"Loading benchmark from: {csv_path}")
        pairwise_df = pd.read_csv(csv_path)
        print(f"  Pairwise samples: {len(pairwise_df)}")

        self.df = convert_pairwise_to_listwise_eval(pairwise_df, config)
        print(f"  Listwise samples: {len(self.df)}")

        self.tokenizer = tokenizer
        self.config = config
        self.use_glove = "glove" in config.model_name.lower()

        # Check if this is LSTUR model
        self.use_lstur = config.architecture == "lstur"
        self.user_to_idx = None

        # Load news_items.csv for text lookup
        print(f"Loading news items from: {news_items_path}")
        self.news_items = pd.read_csv(news_items_path, index_col="item_id")
        print(f"  Loaded {len(self.news_items)} news items")

        # MODIFICATION 2: Load the user map if LSTUR and no map injected
        if self.use_lstur:
            if user_map is not None:
                # Use injected map (from ModelEvaluator)
                self.user_to_idx = user_map
            else:
                # Load from file (standard evaluation case)
                self.user_map_path = self.config.data_dir / "user_to_idx_map.json"
                if self.user_map_path.exists():
                    print(f"  Loading user mapping from: {self.user_map_path}")
                    with open(self.user_map_path, "r") as f:
                        raw_map = json.load(f)
                        # Keys in JSON are strings, convert them back to original type
                        self.user_to_idx = {
                            self._convert_key(k): v for k, v in raw_map.items()
                        }
                else:
                    # If this is the main benchmark run, it MUST find the map.
                    raise FileNotFoundError(
                        f"LSTUR requires a pre-built user map. File not found: {self.user_map_path}"
                    )
            print(f"  Loaded {len(self.user_to_idx)} user entries for LSTUR.")

    def __len__(self):
        return len(self.df)

    def _convert_key(self, key):
        """Helper to convert string keys from JSON back to original type (int or str)."""
        try:
            return int(key)
        except ValueError:
            return key

    def _get_user_idx(self, user_id: str) -> int:
        """Get user index for embedding lookup."""
        if not self.use_lstur:
            return 0

        # Try to convert to appropriate type
        try:
            if isinstance(user_id, str):
                try:
                    user_id = int(user_id)
                except ValueError:
                    pass
        except:
            pass

        # Return mapped index, default to 0 (UNK) if not found
        return self.user_to_idx.get(user_id, 0)

    def _get_text_by_id(self, item_id: str) -> str:
        """Lookup text by item_id from news_items.csv."""
        try:
            return self.news_items.loc[item_id, "text"]
        except KeyError:
            # Fallback to title if text not found
            return self.news_items.loc[item_id, "title"]

    def _get_title_by_id(self, item_id: str) -> str:
        """Lookup title by item_id from news_items.csv."""
        return self.news_items.loc[item_id, "title"]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        impressions = row["impressions"]
        user_id = row["user_id"]
        user_idx = self._get_user_idx(user_id)

        # Sort so positives come first
        impressions.sort(key=lambda x: x[2], reverse=True)

        # Extract candidate IDs and lookup texts and titles
        candidate_ids = [info[0] for info in impressions]
        candidate_texts = [self._get_text_by_id(item_id) for item_id in candidate_ids]
        candidate_titles = [self._get_title_by_id(item_id) for item_id in candidate_ids]

        # Parse history IDs and lookup texts and titles
        try:
            history_ids = ast.literal_eval(row["history_ids"])
        except (ValueError, SyntaxError):
            history_ids = row["history_ids"]

        if not isinstance(history_ids, list):
            history_ids = [str(history_ids)]

        history_ids = history_ids[: self.config.max_history_length]
        history_texts = [self._get_text_by_id(item_id) for item_id in history_ids]
        history_titles = [self._get_title_by_id(item_id) for item_id in history_ids]

        if self.use_glove:
            # For GloVe: return raw text
            return {
                "candidate_texts": candidate_texts,
                "candidate_titles": candidate_titles,
                "history_texts": history_texts,
                "history_titles": history_titles,
                "user_id": user_id,
                "user_ids": torch.tensor(user_idx, dtype=torch.long),  # Add user index
                "impression_data": impressions,
            }
        else:
            # For transformers: tokenize
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for transformer models")

            candidate_text_inputs = self.tokenizer(
                candidate_texts,
                max_length=self.config.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            candidate_title_inputs = self.tokenizer(
                candidate_titles,
                max_length=self.config.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            history_text_inputs = self.tokenizer(
                history_texts,
                max_length=self.config.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            history_title_inputs = self.tokenizer(
                history_titles,
                max_length=self.config.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            return {
                "candidate_input_ids": candidate_text_inputs["input_ids"],
                "candidate_attention_mask": candidate_text_inputs["attention_mask"],
                "candidate_title_input_ids": candidate_title_inputs["input_ids"],
                "candidate_title_attention_mask": candidate_title_inputs[
                    "attention_mask"
                ],
                "history_input_ids": history_text_inputs["input_ids"],
                "history_attention_mask": history_text_inputs["attention_mask"],
                "history_title_input_ids": history_title_inputs["input_ids"],
                "history_title_attention_mask": history_title_inputs["attention_mask"],
                "user_id": user_id,
                "user_ids": torch.tensor(user_idx, dtype=torch.long),  # Add user index
                "impression_data": impressions,
            }


def load_test_data(config, model_config):
    """
    Load test data with fake/real labels.

    Args:
        config: Configuration dict with:
            - data_path: Path to data
            - sample_size: Number of samples per class (None = all)
            - min_history_length: Minimum history length (optional)
            - seed: Random seed for sampling
        model_config: Configuration for model/tokenizer

    Returns:
        pytorch Dataset
    """
    benchmark_real_data_path = Path(config["data_path"] + "benchmark_mixed.csv")
    benchmark_fake_data_path = Path(config["data_path"] + "benchmark_honeypot.csv")

    if benchmark_real_data_path.suffix not in {
        ".csv"
    } and benchmark_fake_data_path.suffix not in {".csv"}:
        raise ValueError(
            "Unsupported data format. Please implement data loading logic."
        )

    if config.get("model_type") == "glove":
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)

    concat_df = pd.concat(
        [pd.read_csv(benchmark_real_data_path), pd.read_csv(benchmark_fake_data_path)],
        ignore_index=True,
    )

    concat_df.to_csv("temp_combined_dataset.csv", index=False)
    data_path = "temp_combined_dataset.csv"

    dataset = BenchmarkDataset(
        csv_path=data_path,
        news_items_path=config.get("news_items_path"),
        tokenizer=tokenizer,
        config=model_config,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=model_config.val_batch_size,
        shuffle=False,
        collate_fn=benchmark_collate_fn,
        num_workers=2,
    )

    return dataset, data_loader


def get_data_statistics_fast(dataset):
    """
    Blazingly fast - directly access the dataframe, avoid __getitem__ entirely.

    Speed: ~1000x faster than original
    Works because: Skips all tokenization/lookup logic
    """
    print("\nComputing data statistics (fast method)...")

    is_fake_list = []
    news_ids = []

    # Direct dataframe access - no __getitem__ calls!
    for idx in tqdm(range(len(dataset.df)), desc="Processing"):
        row = dataset.df.iloc[idx]
        impressions = row["impressions"]

        # Get only the first impression (rank 0)
        if len(impressions) > 0:
            candidate_id, candidate_title, label, is_fake = impressions[0]
            is_fake_list.append(is_fake)
            news_ids.append(candidate_id)

    is_fake_array = np.array(is_fake_list)

    print(f"\nData Statistics:")
    print(f"  Total samples: {len(is_fake_list)}")
    print(
        f"  Fake news: {np.sum(is_fake_array == 1)} ({np.mean(is_fake_array == 1)*100:.1f}%)"
    )
    print(
        f"  Real news: {np.sum(is_fake_array == 0)} ({np.mean(is_fake_array == 0)*100:.1f}%)"
    )
    print(f"  Unique news articles: {len(set(news_ids))}")

    return {
        "is_fake_array": is_fake_array,
        "news_ids": news_ids,
        "n_total": len(is_fake_list),
        "n_fake": np.sum(is_fake_array == 1),
        "n_real": np.sum(is_fake_array == 0),
        "n_unique": len(set(news_ids)),
    }


def get_data_statistics(dataset):
    """
    Print statistics about loaded data from BenchmarkDataset.
    Focus on rank 0 (first candidate) to avoid duplicates.
    """
    is_fake_list = []
    news_ids = []
    clicked_list = []

    # Iterate through the dataset
    for idx in trange(len(dataset)):
        item = dataset[idx]
        impression_data = item["impression_data"]

        # Get only the first impression (rank 0) to avoid duplicates
        if len(impression_data) > 0:
            candidate_id, candidate_title, label, is_fake = impression_data[0]
            is_fake_list.append(is_fake)
            news_ids.append(candidate_id)

    is_fake_array = np.array(is_fake_list)

    print(f"\nData Statistics:")
    print(f"  Total samples: {len(is_fake_list)}")
    print(
        f"  Fake news: {np.sum(is_fake_array == 1)} ({np.mean(is_fake_array == 1)*100:.1f}%)"
    )
    print(
        f"  Real news: {np.sum(is_fake_array == 0)} ({np.mean(is_fake_array == 0)*100:.1f}%)"
    )
    print(f"  Unique news articles: {len(set(news_ids))}")
