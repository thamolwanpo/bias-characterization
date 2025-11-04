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

sys.path.insert(0, os.path.abspath("../plm4newsrs/src/evaluation"))

from benchmark_dataset import BenchmarkDataset, benchmark_collate_fn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


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
        csv_path=data_path, tokenizer=tokenizer, config=model_config
    )

    data_loader = DataLoader(
        dataset,
        batch_size=model_config.val_batch_size,
        shuffle=False,
        collate_fn=benchmark_collate_fn,
        num_workers=2,
    )

    return dataset, data_loader


def get_data_statistics(dataset):
    """
    Print statistics about loaded data from BenchmarkDataset.
    Focus on rank 0 (first candidate) to avoid duplicates.
    """
    is_fake_list = []
    news_ids = []
    clicked_list = []

    # Iterate through the dataset
    for idx in range(len(dataset)):
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
