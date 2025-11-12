
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F

VOCAB="AUGC"; V=4

@dataclass
class CVAEConfig:
    max_len_utr5: int = 512
    max_len_utr3: int = 512
    latent_dim: int = 128
    hidden: int = 512
    num_layers: int = 4
    num_organs: int = 32

class Encoder(nn.Module):
    def __init__(self, cfg: CVAEConfig):
        super().__init__()
        L = cfg.max_len_utr5 + cfg.max_len_utr3
        self.embed = nn.Linear(V, cfg.hidden)
        self.cnn = nn.Sequential(*sum([[nn.Conv1d(cfg.hidden, cfg.hidden, 5, padding=2), nn.GELU()] for _ in range(cfg.num_layers)], []))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.cond = nn.Embedding(cfg.num_organs, cfg.hidden)
        self.mu = nn.Linear(cfg.hidden, cfg.latent_dim)
        self.logvar = nn.Linear(cfg.hidden, cfg.latent_dim)
    def forward(self, xOH, organ_id):
        h = self.embed(xOH.transpose(1,2)).transpose(1,2)
        h = self.cnn(h)
        h = self.pool(h).squeeze(-1) + self.cond(organ_id)
        return self.mu(h), self.logvar(h)

class Decoder(nn.Module):
    def __init__(self, cfg: CVAEConfig):
        super().__init__()
        L = cfg.max_len_utr5 + cfg.max_len_utr3
        self.cond = nn.Embedding(cfg.num_organs, cfg.hidden)
        self.fc = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.hidden), nn.GELU(),
                                nn.Linear(cfg.hidden, cfg.hidden*L), nn.GELU())
        self.proj = nn.Linear(cfg.hidden, V)
        self.L=L; self.hidden=cfg.hidden
    def forward(self, z, organ_id):
        B = z.size(0)
        h = self.fc(z).view(B, self.L, self.hidden) + self.cond(organ_id).unsqueeze(1)
        logits = self.proj(h)  # (B,L,V)
        return logits

class CVAE(nn.Module):
    def __init__(self, cfg: CVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = Encoder(cfg)
        self.dec = Decoder(cfg)
    def forward(self, xOH, organ_id, beta=1.0):
        mu, logvar = self.enc(xOH, organ_id)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        logits = self.dec(z, organ_id)
        recon = F.cross_entropy(logits.view(-1, logits.size(-1)), xOH.argmax(dim=1).view(-1), reduction="mean")
        kl = -0.5*torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta*kl
        return loss, recon.detach(), kl.detach(), logits
