"""Dataset utilities for training the side-feature CNN model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset


def load_manifest(dataset_dir: str) -> Dict[str, Any]:
    """Load the JSON manifest produced by the preprocessing script."""

    manifest_path = Path(dataset_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class UTRFeatureShardDataset(Dataset):
    """Dataset that streams UTR tensors from PyTorch shards with cached loading."""

    def __init__(self, dataset_dir: str, split: str = "train") -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.shards_dir = self.dataset_dir / "shards"
        if not self.shards_dir.exists():
            raise FileNotFoundError(f"Shards directory not found: {self.shards_dir}")
        self.shard_paths: List[Path] = sorted(self.shards_dir.glob("data.part-*.pt"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No .pt shards found under {self.shards_dir}")
        self.index = self._load_index(split)
        if {"part_id", "local_idx"} - set(self.index.columns):
            raise ValueError(f"Index for split '{split}' must contain part_id and local_idx")
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def _load_index(self, split: str) -> pd.DataFrame:
        candidates = [
            self.dataset_dir / "index" / split / "index.parquet",
            self.dataset_dir / f"{split}.index.parquet",
            self.dataset_dir / "index" / split / "index.csv",
            self.dataset_dir / f"{split}.index.csv",
        ]
        for path in candidates:
            if path.exists():
                if path.suffix == ".parquet":
                    return pd.read_parquet(path)
                return pd.read_csv(path)
        raise FileNotFoundError(f"Index file for split '{split}' not found under {self.dataset_dir}")

    def __len__(self) -> int:
        return int(len(self.index))

    def _load_shard(self, part_id: int) -> Dict[str, torch.Tensor]:
        if part_id not in self._cache:
            shard_path = self.shard_paths[part_id]
            data = torch.load(shard_path, map_location="cpu")
            if not isinstance(data, dict):
                raise ValueError(f"Shard {shard_path} must be a dict with tensors")
            self._cache[part_id] = {
                "utr5": data["utr5"].float(),
                "utr3": data["utr3"].float(),
                "organ_id": data["organ_id"].long(),
                "label": data["label"].float(),
            }
        return self._cache[part_id]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.index.iloc[idx]
        part_id = int(row["part_id"])
        local_idx = int(row["local_idx"])
        shard = self._load_shard(part_id)
        utr5 = shard["utr5"][local_idx]
        utr3 = shard["utr3"][local_idx]
        organ = shard["organ_id"][local_idx]
        label = shard["label"][local_idx]
        return {"utr5": utr5, "utr3": utr3, "organ_id": organ, "label": label}
