"""AeroTwin data package.

Exposes the HF-backed loader for remote access to aerotwin/aero-data
(flightlist, fuel labels, per-flight trajectories) without full downloads.
"""

from .loader import (
    AeroDataLoader,
    DATASET_REPO_ID,
    VALID_SPLITS,
    DEFAULT_SPLIT,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_RANDOM_SEED,
)

__all__ = [
    "AeroDataLoader",
    "DATASET_REPO_ID",
    "VALID_SPLITS",
    "DEFAULT_SPLIT",
    "DEFAULT_SAMPLE_SIZE",
    "DEFAULT_RANDOM_SEED",
]
