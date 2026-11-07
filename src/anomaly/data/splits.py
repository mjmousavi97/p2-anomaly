"""Deterministic train/validation split of the normal training images."""
import numpy as np


def split_train_val(n: int, val_fraction: float=0.2, seed:int=42):

    if not 0 < val_fraction < 1:
      raise ValueError("val_fraction must be between 0 and 1")
    
    if n < 2:
      raise ValueError("need at least 2 images to split")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_val = min(max(1, round(n * val_fraction)), n - 1)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])
