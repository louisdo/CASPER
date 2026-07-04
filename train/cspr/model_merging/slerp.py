import os
import shutil
from argparse import ArgumentParser

import yaml
import torch
from safetensors.torch import load_file, save_file

SIDE_FILES = [
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
    "vocab_partitions.json",
    "train_config.json",
]
CONFIG_FILES = ["config.json", "configuration.py"]

# Below this angle (radians) the two tensors are ~colinear; fall back to a plain
# lerp to avoid dividing by sin(w) ~ 0.
_DOT_THRESHOLD = 0.9995


def _slerp_tensor(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """SLERP between two tensors, treating each as a single flattened vector
    (mergekit's default). Falls back to lerp when nearly colinear or when an
    endpoint is ~zero."""
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    af, bf = a.flatten(), b.flatten()
    na, nb = af.norm(), bf.norm()
    if na < 1e-12 or nb < 1e-12:
        return torch.lerp(a, b, t)
    dot = torch.dot(af / na, bf / nb).clamp(-1.0, 1.0)
    if dot.abs() > _DOT_THRESHOLD:
        return torch.lerp(a, b, t)  # nearly colinear -> lerp is numerically safe
    w = torch.acos(dot)
    sin_w = torch.sin(w)
    s0 = torch.sin((1.0 - t) * w) / sin_w
    s1 = torch.sin(t * w) / sin_w
    return s0 * a + s1 * b


def slerp_state_dicts(sd_a: dict[str, torch.Tensor],
                      sd_b: dict[str, torch.Tensor],
                      t: float) -> dict[str, torch.Tensor]:
    ka, kb = set(sd_a), set(sd_b)
    if ka != kb:
        raise ValueError(f"Key mismatch: only_a={list(ka - kb)[:5]} only_b={list(kb - ka)[:5]}")
    return {k: _slerp_tensor(sd_a[k], sd_b[k], t) for k in sd_a}


def _multislerp_tensor(tensors: list[torch.Tensor], weights: list[float],
                       eps: float = 1e-8) -> torch.Tensor:
    """Weighted spherical average (Riemannian barycenter) of N tensors on the
    unit hypersphere, per flattened tensor. Reduces to ordinary SLERP for N=2
    and is symmetric / order-independent for N>2.

    Direction and magnitude are handled separately: directions are averaged on
    the sphere via a one-step log/exp map about their weighted centroid, and the
    output norm is the weighted average of the input norms.
    """
    shape = tensors[0].shape
    flats = [t.to(torch.float32).flatten() for t in tensors]
    norms = torch.stack([f.norm() for f in flats])                 # [N]
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum()

    # Degenerate: any ~zero tensor -> fall back to plain weighted (linear) mean.
    if (norms < eps).any():
        mean = sum(wi * f for wi, f in zip(w, flats))
        return mean.reshape(shape)

    units = torch.stack([f / n for f, n in zip(flats, norms)])     # [N, D] unit dirs

    # Reference direction: normalized weighted mean of the unit vectors.
    ref = (w[:, None] * units).sum(0)
    ref_norm = ref.norm()
    if ref_norm < eps:
        # antipodal / no coherent centroid -> linear mean of directions
        avg_dir = (w[:, None] * units).sum(0)
    else:
        ref = ref / ref_norm
        # log map each unit vector into the tangent space at ref
        dots = (units * ref).sum(1).clamp(-1.0, 1.0)               # [N]
        thetas = torch.acos(dots)                                  # angle to ref
        tangents = units - dots[:, None] * ref                     # component ⊥ ref
        tnorms = tangents.norm(dim=1, keepdim=True)
        logs = torch.where(tnorms > eps,
                           tangents / tnorms * thetas[:, None],
                           torch.zeros_like(tangents))             # [N, D] in tangent
        # weighted average in tangent space, then exp map back to the sphere
        v = (w[:, None] * logs).sum(0)
        vnorm = v.norm()
        if vnorm < eps:
            avg_dir = ref
        else:
            avg_dir = torch.cos(vnorm) * ref + torch.sin(vnorm) * (v / vnorm)

    avg_dir = avg_dir / avg_dir.norm().clamp(min=eps)
    out_norm = (w * norms).sum()                                   # weighted avg magnitude
    return (out_norm * avg_dir).reshape(shape)


def multislerp_state_dicts(state_dicts: list[dict[str, torch.Tensor]],
                           weights: list[float]) -> dict[str, torch.Tensor]:
    ref_keys = set(state_dicts[0])
    for i, sd in enumerate(state_dicts[1:], start=1):
        if set(sd) != ref_keys:
            d = ref_keys.symmetric_difference(set(sd))
            raise ValueError(f"Key mismatch in checkpoint {i}: {list(d)[:5]}")
    return {
        k: _multislerp_tensor([sd[k] for sd in state_dicts], weights)
        for k in state_dicts[0]
    }


def _load(path: str) -> dict[str, torch.Tensor]:
    return load_file(os.path.join(path, "model.safetensors"))


def main():
    ap = ArgumentParser(description="SLERP merge of CSpR checkpoints.")
    ap.add_argument("--config", required=True, help="Path to the YAML SLERP manifest.")
    cli = ap.parse_args()
    with open(cli.config) as f:
        cfg = yaml.safe_load(f)

    parent = cfg["parent_run_dir"]
    output_dir = cfg["output_dir"]
    mode = cfg.get("mode", "pairwise")
    entries = cfg["checkpoints"]

    # t is required for pairwise/chained; multislerp uses per-checkpoint weights.
    t = cfg.get("t")
    if mode in ("pairwise", "chained"):
        if t is None:
            raise SystemExit(f"mode '{mode}' requires 't' in the config.")
        t = float(t)
        if not 0.0 <= t <= 1.0:
            raise SystemExit(f"t must be in [0, 1], got {t}.")

    paths = [e["path"] if os.path.isabs(e["path"]) else os.path.join(parent, e["path"])
             for e in entries]

    if mode == "multislerp":
        if len(paths) < 2:
            raise SystemExit("multislerp needs at least 2 checkpoints.")
        weights = [float(e.get("weight", 1.0)) for e in entries]
        if any(x < 0 for x in weights) or sum(weights) <= 0:
            raise SystemExit(f"multislerp weights must be non-negative and sum > 0: {weights}")
        tot = sum(weights)
        print(f"Multi-SLERP (weighted spherical mean) over {len(paths)} checkpoints:")
        for p, x in zip(paths, weights):
            print(f"  {x / tot:6.3f}  {p}")
        merged = multislerp_state_dicts([_load(p) for p in paths], weights)
    elif mode == "pairwise":
        if len(paths) != 2:
            raise SystemExit("pairwise SLERP needs exactly 2 checkpoints.")
        print(f"SLERP(t={t}) {paths[0]}  ->  {paths[1]}")
        merged = slerp_state_dicts(_load(paths[0]), _load(paths[1]), t)
    elif mode == "chained":
        if len(paths) < 2:
            raise SystemExit("chained SLERP needs at least 2 checkpoints.")
        print(f"Chained SLERP(t={t}) over {len(paths)} checkpoints "
              f"(last listed carries weight t):")
        acc = _load(paths[0])
        print(f"  base:       {paths[0]}")
        for i, p in enumerate(paths[1:], start=1):
            print(f"  slerp #{i}: acc <- slerp(acc, {os.path.basename(p)}, t={t})")
            acc = slerp_state_dicts(acc, _load(p), t)
        merged = acc
    else:
        raise SystemExit(f"Unknown mode '{mode}'. Expected 'pairwise' or 'chained'.")

    os.makedirs(output_dir, exist_ok=True)
    save_file(merged, os.path.join(output_dir, "model.safetensors"))
    print(f"Wrote merged weights ({len(merged)} tensors) -> {output_dir}")

    for f in CONFIG_FILES:
        src = os.path.join(paths[0], f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, f))
    for f in SIDE_FILES:
        src = os.path.join(parent, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, f))
        else:
            print(f"  [warn] side file not found, skipped: {src}")

    print(f"SLERP merge ready: {output_dir}")


if __name__ == "__main__":
    main()
