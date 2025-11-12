
import argparse, json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from src.side.model import DualBranchCNNFiLM
from src.side.dataset import load_manifest

VOCAB = "AUGC"
_tok2id = {c:i for i,c in enumerate(VOCAB)}

def one_hot_encode(seq: str, L: int) -> np.ndarray:
    s = seq.upper().replace("T","U")[-L:]
    s = s + ("U"*(L-len(s)))
    x = np.zeros((4, L), dtype=np.float32)
    for i, ch in enumerate(s):
        j = _tok2id.get(ch, 1)
        x[j, i] = 1.0
    return x

class PairDataset(Dataset):
    def __init__(self, rows: List[Tuple[str,str,int]], L5: int, L3: int):
        self.rows = rows; self.L5=L5; self.L3=L3
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx: int):
        s5, s3, organ = self.rows[idx]
        return {
            "utr5": torch.from_numpy(one_hot_encode(s5, self.L5)),
            "utr3": torch.from_numpy(one_hot_encode(s3, self.L3)),
            "organ_id": torch.tensor(int(organ), dtype=torch.long),
        }

def load_phase1_model(manifest_path: str, ckpt_path: str, device: torch.device):
    man = json.load(open(manifest_path, "r"))
    in_channels = int(man["shapes"]["utr5"][0])
    num_organs = int(man.get("num_organs", 0) or len(man.get("organ_vocab", {})))
    model = DualBranchCNNFiLM(
        in_channels=in_channels, num_organs=num_organs,
        conv_channels=[64,128,256], stem_channels=32, film_dim=32,
        hidden_dim=256, dropout=0.2,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.eval().to(device), man

def infer(model, loader, device):
    preds = []
    with torch.no_grad():
        for batch in loader:
            utr5 = batch["utr5"].to(device)
            utr3 = batch["utr3"].to(device)
            organ = batch["organ_id"].to(device)
            y = model(utr5, utr3, organ)
            preds.append(y.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)

def main():
    import yaml, pandas as pd, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gen_predict.yaml")
    ap.add_argument("--input", required=True, help="CSV/TSV columns: utr5,utr3,organ_id")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    manifest = Path(cfg["dataset_dir"]) / "manifest.json"
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    bs = int(cfg.get("batch_size", 512)); nw = int(cfg.get("num_workers", 0))
    model, man = load_phase1_model(str(manifest), cfg["phase1_checkpoint"], device)
    L5 = int(man["shapes"]["utr5"][1]); L3 = int(man["shapes"]["utr3"][1])

    sep = "," if args.input.endswith(".csv") else "\t"
    df = pd.read_csv(args.input, sep=sep)
    rows = list(zip(df["utr5"].tolist(), df["utr3"].tolist(), df["organ_id"].tolist()))
    ds = PairDataset(rows, L5, L3)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    pred = infer(model, loader, device)
    df["pred"] = pred
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(df)} rows)")

if __name__ == "__main__":
    main()
