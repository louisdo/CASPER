#!/usr/bin/env bash
# SLERP merge of a GTE-MLM CSpR run. Edit checkpoints/t/mode in the manifest.
set -euo pipefail

python -m train.cspr.model_merging.slerp \
    --config train/cspr/model_merging/conf/slerp-gte.yaml
