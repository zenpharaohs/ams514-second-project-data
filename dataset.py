"""Load immutable loss tapes and regroup starts without consulting outcomes."""
from pathlib import Path
import numpy as np

CASES = ("quadratic_gd", "near_saddle_gd", "plateau_gd", "plateau_ngd")
CORPUS = Path(__file__).resolve().parent / "corpus"


def regroup_losses(loss, arms=32):
    """Preserve flat original (pool, arm) order, with disjoint new pools."""
    if loss.ndim != 3:
        raise ValueError("Expected [pool, arm, local_pull] losses")
    total = loss.shape[0] * loss.shape[1]
    if not isinstance(arms, (int, np.integer)) or arms <= 0 or total % arms:
        raise ValueError("arms must be a positive integer dividing the tape bank")
    grouped = loss.reshape(total // arms, arms, loss.shape[2])
    grouped.flags.writeable = False
    return grouped


def load_losses(case="near_saddle_gd", arms=32, corpus=CORPUS):
    """Harness-only loading: policies receive observations, not this array."""
    if case not in CASES:
        raise ValueError(f"Unknown case: {case}")
    with np.load(Path(corpus) / f"{case}.npz", allow_pickle=False) as data:
        return regroup_losses(data["loss"], arms)
