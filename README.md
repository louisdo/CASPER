# CSpR

Retrieval evaluation framework for scientific literature benchmarks.

## Quick Start

```python
from cspr.model import CSpR, to_sparse_dict

vocab_partitions = {
    "token": range(30522),
    "phrase": range(30522, 59419)
}

model = CSpR("lamdo/casper", vocab_partitions=vocab_partitions)

inputs = model.tokenizer("deep transfer learning in neural networks", return_tensors="pt")
encode_out = model.encode(inputs)

sparse_dict = to_sparse_dict(encode_out, vocab_partitions, model.tokenizer)
# {"token": {'deep': 1.429429531097412, 'soap': 1.2387222051620483, ...
#  "phrase": {'deep learning': 2.255920648574829, 'knowledge transfer': 1.355625033378601, ...
```

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
git clone <repo-url>
cd CSpR
```

### Setup environment for dense retrieval

```bash
make venv-dense
```

This creates `.venv-dense` with all dense retrieval dependencies.


### Setup environment for CSpR
TODO

### Setup environment for multi-vector retrieval
TODO



## Dense Retrieval

### Supported models

| Key | HuggingFace model |
|-----|-------------------|
| `specter2` | `allenai/specter2_base` |
| `bge_m3` | `BAAI/bge-m3` |
| `qwen3_embedding` | `Qwen/Qwen3-Embedding-0.6B` |
| `e5_base` | `intfloat/e5-base-v2` |

### Supported datasets

`scifact`, `scidocs`, `nfcorpus`, `doris_mae`, `cfscube`, `acm_cr`, `litsearch`, `relish`


### Data layout

Benchmarks are expected at:

```
{WORK_DIR}/benchmarks/{dataset}/
├── corpus.jsonl
├── queries.jsonl
└── qrels/
    └── test.tsv
```

One can download benchmarks here: TODO: put link here.

### Running

```bash
bash bash_scripts/retriever_evaluation/dense.sh
```

Metrics reported: NDCG, MAP, Recall, Precision, MRR at k = 5, 10, 50, 100, 1000.
