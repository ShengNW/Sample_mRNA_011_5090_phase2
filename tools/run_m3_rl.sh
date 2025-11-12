#!/usr/bin/env bash
set -euo pipefail
python -m src.gen.train_rl --config configs/m3_rl.yaml --predict-config configs/gen_predict.yaml
