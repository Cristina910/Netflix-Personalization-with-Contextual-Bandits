from __future__ import annotations
import numpy as np
import pandas as pd

def compute_ctr(clicks: np.ndarray, window: int = None) -> np.ndarray:
    if window is None:
        cumsum = np.cumsum(clicks)
        t = np.arange(1, len(clicks)+1)
        return cumsum / t
    ctr = np.zeros_like(clicks, dtype=float)
    for i in range(len(clicks)):
        start = max(0, i-window+1)
        ctr[i] = clicks[start:i+1].mean()
    return ctr

def compute_regret(optimal_probs: np.ndarray, taken_probs: np.ndarray) -> np.ndarray:
    """Instantaneous regret, and cumulative sum returned."""
    inst = optimal_probs - taken_probs
    return np.cumsum(inst)

def to_results_df(records):
    """records: list of dict step-wise results"""
    return pd.DataFrame.from_records(records)
