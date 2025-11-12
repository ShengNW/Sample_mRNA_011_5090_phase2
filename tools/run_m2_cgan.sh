#!/usr/bin/env bash
set -euo pipefail
python -m src.gen.train_cgan --config configs/m2_cgan.yaml --seed_csv outputs/phase2/m1/m1_topk.csv
