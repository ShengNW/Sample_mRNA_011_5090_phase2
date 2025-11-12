#!/usr/bin/env bash
set -euo pipefail
python -m src.gen.mutate_search --config configs/m1_search.yaml --predict-config configs/gen_predict.yaml
