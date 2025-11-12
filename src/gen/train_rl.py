
import argparse, yaml, numpy as np, pandas as pd, torch, tempfile, subprocess
from pathlib import Path
from tqdm import trange
from src.gen.rl.ppo import Policy, VOCAB, V

TOK2ID = {c:i for i,c in enumerate(VOCAB)}
ID2TOK = {i:c for c,i in TOK2ID.items()}

def step_env(policy, organ_id, L):
    device = next(policy.parameters()).device
    x = torch.full((1,L), fill_value=TOK2ID["U"], dtype=torch.long, device=device)
    logits, _ = policy(x, torch.tensor([organ_id], device=device))
    probs = torch.softmax(logits, dim=-1)
    x = torch.multinomial(probs.view(-1, V), num_samples=1).view(1, L)
    return x[0].tolist()

def score_seq(predict_cfg, seq5, seq3, organ_id):
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td)/"in.csv"; out = Path(td)/"out.csv"
        pd.DataFrame([{"utr5": seq5, "utr3": seq3, "organ_id": organ_id}]).to_csv(inp, index=False)
        subprocess.run(["python","-m","src.side.predict","--config",predict_cfg,"--input",str(inp),"--out",str(out)], check=True)
        df = pd.read_csv(out)
        return float(df["pred"].iloc[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m3_rl.yaml")
    ap.add_argument("--predict-config", default="configs/gen_predict.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    out_dir = Path(cfg["io"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    L5 = cfg["env"]["max_len_utr5"]; L3=cfg["env"]["max_len_utr3"]; L=L5+L3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy(L5,L3, hidden=512, num_organs=128).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg["train"]["lr"])
    target = int(cfg["env"]["target_organ"])
    for step in trange(cfg["train"]["steps"], desc="RL"):
        x_idx = torch.tensor([step_env(policy, target, L)], device=device)
        seq = "".join(ID2TOK[int(i)] for i in x_idx[0].tolist())
        seq5, seq3 = seq[:L5], seq[L5:]
        r = score_seq(args.predict_config, seq5, seq3, target)
        logits, value = policy(x_idx, torch.tensor([target], device=device))
        logp = torch.log_softmax(logits, dim=-1)
        chosen = logp[0, torch.arange(L), x_idx[0]]
        loss = -chosen.mean()*r + (value- r)**2
        opt.zero_grad(); loss.backward(); opt.step()
        if (step+1)%1000==0:
            with open(out_dir/"samples.csv","a") as fh:
                fh.write(f"{seq5},{seq3},{target},{r}\\n")
    torch.save(policy.state_dict(), out_dir/"ppo_final.pt")
    print(f"[RL] Done. Saved policy and samples in {out_dir}")

if __name__ == "__main__":
    main()
