import torch
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

MODEL_HF_IDS = {
    "specter2": "allenai/specter2_base",
    "bge_m3": "BAAI/bge-m3",
    "qwen3_embedding": "Qwen/Qwen3-Embedding",
    "e5_base": "intfloat/e5-base",
}

# prefix_["query"] prepended during retrieval, prefix_["doc"] during indexing.
# None means no prefix for that side.
MODEL_PREFIXES = {
    "specter2": None,
    "bge_m3": None,
    "qwen3_embedding": {
        "query": "Instruct: Given a scientific query, retrieve relevant passages that answer the query\nQuery:",
        "doc":   None,
    },
    "e5_base": {"query": "query:", "doc": "passage:"},
}


@dataclass
class DenseSearchResult:
    docid: str
    score: float


def init_model(model_name):
    hf_id = MODEL_HF_IDS[model_name]
    trust_remote = model_name == "qwen3_embedding"
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust_remote)
    model = AutoModel.from_pretrained(hf_id, trust_remote_code=trust_remote)
    if model_name == "qwen3_embedding":
        # decoder model — left-pad so last token pooling lands on the right position
        tokenizer.padding_side = "left"
    return model, tokenizer, MODEL_PREFIXES.get(model_name)


def _last_token_pool(last_hidden_state, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]


def text_embedding_batch(batch, model, tokenizer, model_name, prefix=None):
    if prefix is not None:
        batch = [prefix + " " + text for text in batch]
    inputs = tokenizer(batch, padding=True, truncation=True,
                       return_tensors="pt", return_token_type_ids=False, max_length=256).to(DEVICE)
    output = model(**inputs)

    if model_name in ["specter2", "bge_m3"]:
        return output.last_hidden_state[:, 0, :].cpu()

    elif model_name in ["e5_base"]:
        attention_mask = inputs["attention_mask"]
        last_hidden = output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return (last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]).cpu()

    elif model_name in ["qwen3_embedding"]:
        return _last_token_pool(output.last_hidden_state, inputs["attention_mask"]).cpu()

    else:
        raise NotImplementedError
