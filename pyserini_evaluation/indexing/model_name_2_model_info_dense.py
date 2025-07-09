from adapters import AutoAdapterModel
from transformers import AutoTokenizer, AutoModel

model_name_2_model_path = {
    "specter2": "allenai/specter2_base",
    "e5_base": "intfloat/e5-base-v2"
}

model_name_2_model_class = {
    "specter2": AutoAdapterModel,
    "e5_base": AutoModel
}

model_name_2_tokenizer_class = {
    "specter2": AutoTokenizer,
    "e5_base": AutoTokenizer
}

model_name_2_prefix = {
    "e5_base": {"query": "query:", "doc": "document:"}
}