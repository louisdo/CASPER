# CSpR

Retrieval evaluation framework for scientific literature benchmarks.

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
