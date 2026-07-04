#!/usr/bin/env bash
# Multi-SLERP merge of a GTE-MLM CSpR run (weighted spherical mean of all
# checkpoints at once). Edit checkpoints/weights in the manifest.
set -euo pipefail

python -m train.cspr.model_merging.slerp \
    --config train/cspr/model_merging/conf/multislerp-gte.yaml
