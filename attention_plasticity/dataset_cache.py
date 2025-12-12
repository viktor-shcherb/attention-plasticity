"""Utilities for downloading datasets before analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download


def prepare_local_dataset(
    repo_id: str,
    model_dir: str,
    cache_dir: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> Path:
    """
    Download all parquet shards for `model_dir` in a single snapshot_download call.

    Returns the local root directory that mirrors the dataset repo structure.
    Repeated calls reuse the cached snapshot inside `cache_dir` (or the default
    Hugging Face cache) without re-downloading unchanged files.
    """
    allow_pattern = f"{model_dir}/**/*.parquet"
    kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "allow_patterns": allow_pattern,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if max_workers is not None:
        kwargs["max_workers"] = max(1, int(max_workers))
    path = snapshot_download(**kwargs)
    return Path(path)
