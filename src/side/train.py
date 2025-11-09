"""Training entry-point for the FiLM-conditioned CNN."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml

from src.side.dataset import UTRFeatureShardDataset, load_manifest
from src.side.model import DualBranchCNNFiLM


def setup_device() -> Tuple[torch.device, bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    return device, distributed, rank, local_rank, world_size


def create_dataloaders(
    dataset_dir: str,
    batch_size: int,
    num_workers: int,
    distributed: bool,
    rank: int,
    world_size: int,
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val"):
        try:
            dataset = UTRFeatureShardDataset(dataset_dir, split=split)
        except FileNotFoundError:
            if split == "val":
                continue
            raise
        sampler = None
        if distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=(split == "train"),
            )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None and split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler,
        )
    return loaders


def compute_r2(preds: np.ndarray, targets: np.ndarray) -> float:
    if preds.size == 0:
        return float("nan")
    ss_res = np.sum((preds - targets) ** 2)
    mean_y = np.mean(targets)
    ss_tot = np.sum((targets - mean_y) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def run_training(cfg: Dict) -> None:
    device, distributed, rank, local_rank, world_size = setup_device()
    dataset_dir = cfg["dataset_dir"]
    manifest = load_manifest(dataset_dir)
    organ_vocab = manifest.get("organ_vocab", {})
    num_organs = max(len(organ_vocab), int(cfg.get("num_organs", 0)))
    if num_organs == 0:
        raise ValueError("Number of organs could not be determined from manifest or config")
    input_channels = manifest["shapes"]["utr5"][0]
    loaders = create_dataloaders(
        dataset_dir,
        batch_size=cfg.get("batch_size", 64),
        num_workers=cfg.get("num_workers", 4),
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    if "train" not in loaders:
        raise RuntimeError("Training split not found in dataset")

    model = DualBranchCNNFiLM(
        in_channels=input_channels,
        num_organs=num_organs,
        conv_channels=cfg.get("conv_channels", [64, 128, 256]),
        stem_channels=cfg.get("stem_channels", 32),
        film_dim=cfg.get("film_dim", 32),
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("learning_rate", 1e-3), weight_decay=cfg.get("weight_decay", 1e-4))

    best_r2 = -float("inf")
    best_state = None
    epochs = cfg.get("epochs", 20)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        samples = 0
        for batch in loaders["train"]:
            utr5 = batch["utr5"].to(device, non_blocking=True)
            utr3 = batch["utr3"].to(device, non_blocking=True)
            organ = batch["organ_id"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            preds = model(utr5, utr3, organ)
            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * target.size(0)
            samples += target.size(0)
        train_loss /= max(samples, 1)
        if rank == 0:
            print(f"Epoch {epoch}/{epochs} - train loss: {train_loss:.4f}")

        if "val" in loaders:
            model.eval()
            preds_all: List[float] = []
            labels_all: List[float] = []
            with torch.no_grad():
                for batch in loaders["val"]:
                    utr5 = batch["utr5"].to(device, non_blocking=True)
                    utr3 = batch["utr3"].to(device, non_blocking=True)
                    organ = batch["organ_id"].to(device, non_blocking=True)
                    target = batch["label"].to(device, non_blocking=True)
                    outputs = model(utr5, utr3, organ)
                    preds_all.append(outputs.detach().cpu().numpy())
                    labels_all.append(target.detach().cpu().numpy())
            if preds_all:
                preds_arr = np.concatenate(preds_all)
                labels_arr = np.concatenate(labels_all)
                r2 = compute_r2(preds_arr, labels_arr)
                if rank == 0:
                    print(f"  Validation R2: {r2:.4f}")
                if r2 > best_r2 and rank == 0:
                    best_r2 = r2
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if rank == 0 and best_state is not None:
        out_path = Path(cfg.get("output_model_path", "outputs/best_model.pt"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, out_path)
        print(f"Saved best model to {out_path} (R2={best_r2:.4f})")
    if distributed:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the FiLM CNN with UTR features")
    parser.add_argument("--config", required=True, help="Path to YAML configuration")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    run_training(cfg)


if __name__ == "__main__":
    main()
