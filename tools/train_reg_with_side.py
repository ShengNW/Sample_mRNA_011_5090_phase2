# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, time, argparse, random
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Sampler
import torch.nn as nn
import torch.optim as optim

from src.cnn_v1.dataset import ShardedUTRDataset
from src.cnn_v1.dataset_side import AugmentedUTRDataset, collate_with_side
from src.cnn_v1.model import DualBranchCNN
from src.cnn_v1.model_side import DualBranchCNNWithSide
from src.features.store import FeatureStore

# 栈快照（卡住时： kill -USR1 <PID>）
import faulthandler, signal
faulthandler.enable(all_threads=True)
faulthandler.register(signal.SIGUSR1, file=open("stack.txt","w"), all_threads=True)

def load_manifest(dataset_dir: str):
    path = os.path.join(dataset_dir, "manifest.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------- 用“局部索引”按分片成组的 BatchSampler --------
class PartGroupedBatchSampler(Sampler):
    """
    给定每个样本所属分片的 part_id（针对“当前 split”），把同一分片内的样本尽量凑成 batch。
    采样器产出的索引是“当前 split 的局部索引”，与 Dataset.__getitem__ 对齐。
    """
    def __init__(self, part_ids, batch_size, drop_last=True, shuffle_groups=True, seed=42):
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.shuffle_groups = shuffle_groups
        self.rng = random.Random(seed)

        # part_id -> [局部索引...]
        buckets = defaultdict(list)
        for i, p in enumerate(part_ids):
            buckets[int(p)].append(i)

        parts = list(buckets.keys())
        if self.shuffle_groups:
            self.rng.shuffle(parts)

        self.batches = []
        for p in parts:
            lst = buckets[p]
            if self.shuffle_groups:
                self.rng.shuffle(lst)
            for i in range(0, len(lst), self.batch_size):
                chunk = lst[i:i+self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                self.batches.append(chunk)

        if self.shuffle_groups:
            self.rng.shuffle(self.batches)

    def __iter__(self):
        for b in self.batches:
            yield b

    def __len__(self):
        return len(self.batches)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="data/processed/seq_cnn_v1_reg")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--batch_size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--huber_beta", type=float, default=1.0)
    ap.add_argument("--logdir", default="outputs/cnn_v1_reg_side")
    # DataLoader 控制
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--prefetch", type=int, default=2)
    ap.add_argument("--pin-memory", type=int, default=1)
    ap.add_argument("--persistent-workers", type=int, default=1)
    ap.add_argument("--print-every", type=int, default=200)
    # 采样策略
    ap.add_argument("--sampler", choices=["part","default"], default="part")  # ← 默认按分片分组
    ap.add_argument("--drop-last", type=int, default=1)
    # 侧表
    ap.add_argument("--tissue", default=None)
    ap.add_argument("--rbp", default=None)
    ap.add_argument("--struct", default=None)
    ap.add_argument("--organ-id-type", default="auto", choices=["auto","int","str"])
    args = ap.parse_args()

    print(f"[debug] PID={os.getpid()}  卡住时执行： kill -USR1 {os.getpid()}")
    os.makedirs(args.logdir, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    fs = FeatureStore(tissue_path=args.tissue, rbp_path=args.rbp, struct_path=args.struct,
                      organ_id_type=args.organ_id_type)

    man = load_manifest(args.dataset_dir)
    out_dim = int(man.get("y_dim", man.get("num_outputs", 54)))
    part_sizes = [int(p["size"]) for p in man.get("parts", [])]  # 仍保留（如需统计/日志使用）

    def build_loader(split):
        base = ShardedUTRDataset(args.dataset_dir, split=split)
        # 优先 JSON 索引；存在就用（避免 parquet 的慢路径）
        index_json = os.path.join(args.dataset_dir, "index", f"{split}.json")
        index_json = index_json if os.path.exists(index_json) else None
        aux = AugmentedUTRDataset(base, fs, index_json=index_json)

        if args.sampler == "part":
            # 直接用“当前 split”的分片标记：长度 == len(base)
            # 注意：这是 dataset 的内部数组，正好就是局部索引 -> 分片 id 的映射
            part_ids = list(map(int, getattr(base, "_idx_part")))
            bs = PartGroupedBatchSampler(
                part_ids=part_ids,
                batch_size=args.batch_size,
                drop_last=bool(args.drop_last),
                shuffle_groups=True,
                seed=1234,
            )
            return DataLoader(
                aux,
                batch_sampler=bs,
                num_workers=args.num_workers,
                pin_memory=bool(args.pin_memory),
                prefetch_factor=args.prefetch if args.num_workers > 0 else None,
                persistent_workers=bool(args.persistent_workers) if args.num_workers > 0 else False,
                collate_fn=collate_with_side,
            )
        else:
            return DataLoader(
                aux,
                batch_size=args.batch_size,
                shuffle=(split=="train"),
                num_workers=args.num_workers,
                pin_memory=bool(args.pin_memory),
                prefetch_factor=args.prefetch if args.num_workers > 0 else None,
                persistent_workers=bool(args.persistent_workers) if args.num_workers > 0 else False,
                collate_fn=collate_with_side,
            )

    loaders = {s: build_loader(s) for s in args.splits.split(",")}

    base = DualBranchCNN(in_ch=5, emb_dim=8, channels=[64,128,256], num_classes=out_dim)
    dims = fs.dims
    print(f"[Side] dims: {dims} (organ_id_type={args.organ_id_type})")
    model = DualBranchCNNWithSide(base, side_dims=dims, out_dim=out_dim, hidden=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    crit = nn.HuberLoss(delta=args.huber_beta)
    opt  = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # warmup：拿到首个 batch
    wb = next(iter(loaders["train"]))
    print(f"[warmup] got first batch: x5={tuple(wb['utr5'].shape)} x3={tuple(wb['utr3'].shape)}")

    best_val = float("inf")
    for epoch in range(1, args.epochs+1):
        model.train()
        tot = 0.0; n = 0; step = 0
        t0 = time.time()
        for batch in loaders["train"]:
            step += 1
            x5 = batch["utr5"].to(device, non_blocking=True).float()
            x3 = batch["utr3"].to(device, non_blocking=True).float()
            y  = batch["label"].to(device, non_blocking=True).float()
            side = {k: v.to(device, non_blocking=True).float() for k,v in batch.get("side", {}).items()}

            opt.zero_grad(set_to_none=True)
            yhat = model(x5, x3, side)
            loss = crit(yhat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot += float(loss.item()) * y.size(0); n += y.size(0)

            if step % max(1, args.print_every) == 0:
                dt = time.time() - t0
                sps = (args.print_every*args.batch_size)/max(dt,1e-6)
                print(f"[train] step={step} dt={dt:.2f}s ~{sps:.1f} samples/s loss={loss.item():.4f}")
                t0 = time.time()
        train_loss = tot / max(n,1)

        # val
        model.eval()
        with torch.no_grad():
            tot = 0.0; n=0
            for batch in loaders.get("val", []):
                x5 = batch["utr5"].to(device, non_blocking=True).float()
                x3 = batch["utr3"].to(device, non_blocking=True).float()
                y  = batch["label"].to(device, non_blocking=True).float()
                side = {k: v.to(device, non_blocking=True).float() for k,v in batch.get("side", {}).items()}
                yhat = model(x5, x3, side)
                loss = crit(yhat, y)
                tot += float(loss.item()) * y.size(0); n += y.size(0)
            val_loss = tot / max(n,1) if n>0 else float("nan")
        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    # （可选）测试略
    # ...

if __name__ == "__main__":
    main()
