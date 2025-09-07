"""Normalization utilities for VLA datasets.

This module provides classes for computing and applying normalization statistics,
adapted from OpenPI but compatible with our PyTorch-based data pipeline.
"""

import json
import pathlib
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class NormStats:
    """Normalization statistics (compatible with OpenPI format)."""
    mean: np.ndarray
    std: np.ndarray
    q01: Optional[np.ndarray] = None  # 1st quantile
    q99: Optional[np.ndarray] = None  # 99th quantile


class RunningStats:
    """Compute running statistics of a batch of vectors (adapted from OpenPI)."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch) -> None:
        """Update the running statistics with a batch of vectors."""
        # Handle torch tensors
        if torch.is_tensor(batch):
            batch = batch.detach().cpu().numpy()
        elif not isinstance(batch, np.ndarray):
            batch = np.array(batch)
        
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        """Compute and return the statistics as a NormStats object."""
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(
            mean=self._mean,
            std=stddev,
            q01=q01,
            q99=q99,
        )

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


def _norm_stats_to_dict(norm_stats: NormStats) -> Dict[str, Any]:
    """Convert NormStats object to dictionary for JSON serialization."""
    return {
        "mean": norm_stats.mean.tolist(),
        "std": norm_stats.std.tolist(),
        "q01": norm_stats.q01.tolist() if norm_stats.q01 is not None else None,
        "q99": norm_stats.q99.tolist() if norm_stats.q99 is not None else None,
    }


def _dict_to_norm_stats(data: Dict[str, Any]) -> NormStats:
    """Convert dictionary back to NormStats object."""
    return NormStats(
        mean=np.array(data["mean"]),
        std=np.array(data["std"]),
        q01=np.array(data["q01"]) if data.get("q01") is not None else None,
        q99=np.array(data["q99"]) if data.get("q99") is not None else None,
    )


def save_stats(norm_stats: Dict[str, NormStats], output_path: Path) -> None:
    """Save normalization stats to a JSON file in OpenPI-compatible format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert NormStats objects to dictionaries for JSON serialization
    serializable_stats = {
        key: _norm_stats_to_dict(stats) for key, stats in norm_stats.items()
    }
    
    with open(output_path, 'w') as f:
        json.dump(serializable_stats, f, indent=2)


def load_stats(stats_path: Path) -> Dict[str, NormStats]:
    """Load normalization stats from a JSON file in OpenPI format."""
    if not stats_path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {stats_path}")
    
    with open(stats_path, 'r') as f:
        data = json.load(f)
    
    # Convert dictionaries back to NormStats objects
    return {
        key: _dict_to_norm_stats(stats_dict) for key, stats_dict in data.items()
    }


def save_stats_openpi_style(norm_stats: Dict[str, NormStats], directory: Path) -> None:
    """Save normalization stats to a directory in OpenPI's exact format."""
    stats_path = directory / "norm_stats.json" 
    save_stats(norm_stats, stats_path)


def load_stats_openpi_style(directory: Path) -> Dict[str, NormStats]:
    """Load normalization stats from a directory in OpenPI's format."""
    stats_path = directory / "norm_stats.json"
    return load_stats(stats_path)
