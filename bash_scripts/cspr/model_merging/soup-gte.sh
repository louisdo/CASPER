#!/usr/bin/env bash
# Weighted model soup of a GTE-MLM CSpR run. Edit checkpoints/weights in the
# manifest, not here.
set -euo pipefail

python -m train.cspr.model_merging.soup \
    --config train/cspr/model_merging/conf/soup-gte.yaml
