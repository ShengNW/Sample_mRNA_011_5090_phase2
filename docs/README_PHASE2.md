# Phase 2 (M1 → M2 → M3) — Drop‑in Scaffold

Merge this into your repo root: `/root/autodl-tmp/Sample_mRNA_011_5090_phase2/src_phase2`.

Adds:
- `src/side/predict.py` — batch scorer loading Phase1 weights to score candidate UTR pairs.
- `src/gen/mutate_search.py` — **M1**: mutation + recombination search with constraints.
- `src/gen/models/cvae.py`, `src/gen/train_cvae.py` — **M2‑A**.
- `src/gen/models/cgan.py`, `src/gen/train_cgan.py` — **M2‑B**.
- `src/gen/rl/ppo.py`, `src/gen/train_rl.py` — **M3**.
- Configs + shell launchers under `configs/` and `tools/`.
