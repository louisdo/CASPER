"""Collect the venue vocabulary from raw S2ORC shards.

Scans every ``*.gz`` shard and extracts venue strings from both the record's own
venue (the ``venue`` annotation) and its cited-reference venues (the ``bibvenue``
annotation), pooled together. No normalization is applied -- raw venue strings
are collected verbatim with occurrence counts:

    { "<raw venue string>": <count>, ... }

Canonicalization / grouping happens in a later step; this pass only gathers the
surface forms.

Shards are processed in parallel (one worker per shard); per-shard partial
counts are merged into the final JSON.

Run:
    python -m data_processing.venue_grouping.collect_venues \
        --shard-dir /scratch/s2orc-20251016 \
        --output /scratch/lamdo/CSpR/venue_grouping/venue_counts.json \
        --num-workers 16
"""
import gzip
import json
import glob
import os
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm


def _get_spans(annotations, ann_type):
    raw = (annotations or {}).get(ann_type)
    if not raw:
        return []
    return json.loads(raw) if isinstance(raw, str) else raw


def _slice(text, span):
    """Slice text[start:end], tolerating string offsets and malformed spans.
    Returns None if the span can't be resolved to a valid substring."""
    try:
        start = int(span["start"])
        end = int(span["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if start < 0 or end > len(text) or start >= end:
        return None
    return text[start:end]


def _venue_strings(record):
    """All venue strings for one record: own venue + cited bibvenues, pooled."""
    content = record.get("content") or {}
    text = content.get("text") or ""
    ann = content.get("annotations") or {}
    out = []
    for ann_type in ("venue", "bibvenue"):
        for s in _get_spans(ann, ann_type):
            v = _slice(text, s)
            if v is not None:
                v = v.strip()
                if v:
                    out.append(v)
    return out


def process_shard(shard_path: str):
    """Return a Counter: raw_venue_string -> count for one shard."""
    local = Counter()
    try:
        with gzip.open(shard_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                local.update(_venue_strings(record))
    except (EOFError, OSError) as exc:
        return {"__error__": f"{shard_path}: {exc}"}
    return dict(local)


def main():
    ap = ArgumentParser(description="Collect raw venue strings from S2ORC shards.")
    ap.add_argument("--shard-dir", required=True, help="Directory of *.gz S2ORC shards.")
    ap.add_argument("--output", required=True, help="Output venue counts JSON.")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit-shards", "--limit", type=int, default=0, dest="limit_shards",
                    help="Process only the first N shards (smoke test). 0 = all.")
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.shard_dir, "*.gz")))
    if args.limit_shards > 0:
        shards = shards[:args.limit_shards]
    if not shards:
        raise SystemExit(f"No *.gz shards found in {args.shard_dir}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Processing {len(shards)} shards with {args.num_workers} workers.")

    acc = Counter()
    errors = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(process_shard, s): s for s in shards}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="shards"):
            partial = fut.result()
            if "__error__" in partial:
                errors.append(partial["__error__"])
                continue
            acc.update(partial)

    # sorted by count, descending -> head venues first
    out = dict(acc.most_common())
    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"Done. {len(out)} distinct venue strings -> {args.output}")
    print(f"  total venue occurrences: {sum(out.values())}")
    if errors:
        print(f"  [warn] {len(errors)} shard(s) failed:")
        for e in errors[:10]:
            print(f"    {e}")


if __name__ == "__main__":
    main()
