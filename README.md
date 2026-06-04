# Installation

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- CUDA-capable GPU (recommended)

## Setup

**1. Clone the repository**

```bash
git clone <repo-url>
cd CSpR
```

**2. Create a virtual environment and install dependencies**

```bash
uv venv --python 3.10
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

**3. Install required packages**

```bash
uv pip install -r requirements.txt
```