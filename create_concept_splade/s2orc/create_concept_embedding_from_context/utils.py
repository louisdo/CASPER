from transformers import AutoTokenizer, AutoModel
import torch


BERT = {
    "model": None,
    "tokenizer": None,
    "model_name": None
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_LENGTH = 256


def init_bert(model_name):
    if model_name != BERT["model_name"]:
        print(f"Initialize BERT: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        model = model.to(DEVICE)

        BERT["model"] = model
        BERT["tokenizer"] = tokenizer
        BERT["model_name"] = model_name


def get_mask_embeddings(texts: list):

    assert len(texts) <= 256, "To prevent OOM"
    
    inputs = BERT["tokenizer"](texts, return_tensors='pt', padding=True, truncation=True, max_length = MAX_LENGTH).to(DEVICE)
    
    mask_token_indices = (inputs.input_ids == BERT["tokenizer"].mask_token_id).int().argmax(dim=1)
    
    # Forward pass
    with torch.no_grad():
        outputs = BERT["model"](**inputs)
    
    # Extract hidden states for mask positions
    hidden_states = outputs.last_hidden_state
    batch_indices = torch.arange(hidden_states.size(0))  # [0, 1, ..., batch_size-1]
    mask_embeddings = hidden_states[batch_indices, mask_token_indices]
    
    return mask_embeddings


if __name__ == "__main__":
    texts = [
        "The capital of France is [MASK].",
        "A [MASK] is a popular pet animal.",
        "2 + 2 equals [MASK]."
    ]

    init_bert("distilbert-base-uncased")
    
    embeddings = get_mask_embeddings(texts)
    print(f"Embeddings shape: {embeddings.shape}")  # Should be (3, 768) for BERT-base