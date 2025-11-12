#!/usr/bin/env bash
set -euo pipefail
python -m src.gen.train_cvae --config configs/m2_cvae.yaml --train_csv outputs/phase2/m1/m1_topk.csv
