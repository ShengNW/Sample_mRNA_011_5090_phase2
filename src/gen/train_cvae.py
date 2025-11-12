
import argparse, yaml
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.gen.models.cvae import CVAE, CVAEConfig, VOCAB

TOK2ID = {c:i for i,c in enumerate(VOCAB)}

def one_hot_batch(seqs, L):
    B = len(seqs)
    x = torch.zeros(B, len(VOCAB), L, dtype=torch.float32)
    for i,s in enumerate(seqs):
        s = s.upper().replace("T","U")
        s = s[:L].ljust(L, "U")
        for j,ch in enumerate(s):
            x[i, TOK2ID.get(ch,1), j] = 1.0
    return x

class SeqPairs(Dataset):
    def __init__(self, df, L5, L3):
        self.df=df; self.L5=L5; self.L3=L3
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        s5, s3 = r.utr5, r.utr3
        x = torch.cat([one_hot_batch([s5], self.L5), one_hot_batch([s3], self.L3)], dim=-1).squeeze(0)
        organ = int(r.organ_id)
        return x, organ

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2_cvae.yaml")
    ap.add_argument("--train_csv", default="outputs/phase2/m1/m1_topk.csv")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    out_dir = Path(cfg["io"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    L5 = cfg["arch"]["max_len_utr5"]; L3 = cfg["arch"]["max_len_utr3"]
    df = pd.read_csv(args.train_csv)
    ds = SeqPairs(df, L5, L3)
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=2)

    num_organs = int(df["organ_id"].max())+1 if "organ_id" in df.columns else 32
    mcfg = CVAEConfig(max_len_utr5=L5, max_len_utr3=L3, latent_dim=cfg["arch"]["latent_dim"], hidden=cfg["arch"]["hidden"],
                      num_layers=cfg["arch"]["num_layers"], num_organs=num_organs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])

    global_step=0
    for epoch in range(cfg["train"]["epochs"]):
        for xb, org in tqdm(dl, desc=f"CVAE epoch {epoch+1}"):
            xb = xb.to(device); org = org.to(device)
            beta = cfg["train"].get("kl_anneal", 0.0003) * (1+global_step)
            loss, recon, kl, _ = model(xb, org, beta=beta)
            opt.zero_grad(); loss.backward(); opt.step()
            global_step += 1
        torch.save(model.state_dict(), out_dir/f"cvae_epoch{epoch+1}.pt")
    print(f"[CVAE] Done. Weights under {out_dir}")

if __name__ == "__main__":
    main()
