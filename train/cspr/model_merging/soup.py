"""Model soup for CSpR checkpoints: weighted linear average of the
``model.safetensors`` weights across several checkpoints of the SAME run
(same architecture, same tokenizer / vocab partitions).

The checkpoints on disk are plain ``NewForMaskedLM`` (GTE) or DistilBERT MLM
safetensors -- the ``PartitionedMaskedLM`` wrapper is only applied at load time
in ``train/cspr/model.py``. So souping just means averaging the raw tensors and
re-assembling a complete, loadable checkpoint directory:

  * weights          -> weighted mean of each checkpoint's model.safetensors
  * config.json      -> copied from the first checkpoint
  * configuration.py -> copied from the first checkpoint (custom modeling code)
  * tokenizer / partition side files -> copied from the parent run dir

The result loads through the normal ``CSpR(model_type_or_dir=..., trust_remote_code=True)``
path with no code changes.

The soup is described by a YAML manifest (see train/cspr/model_merging/conf/):

    parent_run_dir: /scratch/.../2026-06-30_07AM
    output_dir: /scratch/.../2026-06-30_07AM/soup_16-20k_wlinear
    checkpoints:
      - path: checkpoint-16000   # relative to parent_run_dir, or absolute
        weight: 1
      - path: checkpoint-20000
        weight: 5

Run:
    python -m train.cspr.model_merging.soup --config train/cspr/model_merging/conf/soup-gte.yaml
"""
import os
import shutil
from argparse import ArgumentParser

import yaml
import torch
from safetensors.torch import load_file, save_file

# Side files that live in the parent run dir (not in each checkpoint-*) and are
# required to load the model through the CSpR tokenizer/partition pipeline.
SIDE_FILES = [
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
    "vocab_partitions.json",
    "train_config.json",
]
# Copied from a source checkpoint (define the architecture / custom modeling).
CONFIG_FILES = ["config.json", "configuration.py"]


def average_state_dicts(paths: list[str],
                        weights: list[float]) -> dict[str, torch.Tensor]:
    """Weighted mean of the safetensors state dicts, computed incrementally so
    only one checkpoint is held in memory at a time.

    ``weights`` gives the relative importance of each checkpoint (same order as
    ``paths``); they are normalized to sum to 1, so the result is a convex
    combination regardless of scale.
    """
    n = len(paths)
    if len(weights) != n:
        raise ValueError(f"Got {len(weights)} weights for {n} checkpoints.")
    if any(w < 0 for w in weights):
        raise ValueError(f"Weights must be non-negative, got {weights}.")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Weights sum to zero.")
    norm = [w / total for w in weights]

    acc: dict[str, torch.Tensor] = {}
    ref_keys: set[str] | None = None
    for p, w in zip(paths, norm):
        sd = load_file(os.path.join(p, "model.safetensors"))
        keys = set(sd.keys())
        if ref_keys is None:
            ref_keys = keys
        elif keys != ref_keys:
            missing, extra = ref_keys - keys, keys - ref_keys
            raise ValueError(
                f"Key mismatch in {p}: missing={list(missing)[:5]} extra={list(extra)[:5]}"
            )
        for k, v in sd.items():
            t = v.to(torch.float32) * w
            acc[k] = t if k not in acc else acc[k] + t
    return acc


def main():
    ap = ArgumentParser(description="Average CSpR checkpoints into a model soup.")
    ap.add_argument("--config", required=True,
                    help="Path to the YAML soup manifest.")
    cli = ap.parse_args()
    with open(cli.config) as f:
        cfg = yaml.safe_load(f)

    parent = cfg["parent_run_dir"]
    output_dir = cfg["output_dir"]
    entries = cfg["checkpoints"]
    if len(entries) < 2:
        raise SystemExit("Need at least 2 checkpoints to soup.")

    # path is relative to parent_run_dir unless absolute; weight defaults to 1.
    paths = [e["path"] if os.path.isabs(e["path"]) else os.path.join(parent, e["path"])
             for e in entries]
    weights = [float(e.get("weight", 1.0)) for e in entries]

    os.makedirs(output_dir, exist_ok=True)
    total = sum(weights)
    print(f"Averaging {len(paths)} checkpoints -> {output_dir}")
    for p, w in zip(paths, weights):
        print(f"  {w / total:6.3f}  {p}")

    averaged = average_state_dicts(paths, weights)
    save_file(averaged, os.path.join(output_dir, "model.safetensors"))
    print(f"Wrote averaged weights ({len(averaged)} tensors).")

    # config / custom modeling code from the first checkpoint
    for f in CONFIG_FILES:
        src = os.path.join(paths[0], f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, f))
    # tokenizer + partition side files from the parent run dir
    for f in SIDE_FILES:
        src = os.path.join(parent, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, f))
        else:
            print(f"  [warn] side file not found, skipped: {src}")

    print(f"Soup ready: {output_dir}")


if __name__ == "__main__":
    main()
