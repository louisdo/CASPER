import os, json
from collections import defaultdict

def _load_keyphrase_index(path: str) -> dict:
    if os.path.isdir(path):
        keyphrase_index = defaultdict(list)
        for fname in os.listdir(path):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(path, fname)) as f:
                for kp, docs in json.load(f).items():
                    keyphrase_index[kp].extend(docs)
        return dict(keyphrase_index)
    else:
        with open(path) as f:
            return json.load(f)