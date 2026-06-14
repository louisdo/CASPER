import json, os, torch
from typing import Dict, List, Tuple
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

from pmb import PartitionedModernBertForMaskedLM

def load_wiki_definition(path: str) -> Dict[str, str]:
    with open(path) as f:
        definition = json.load(f)

    return definition

def load_keyphrase_vocab(path: str) -> List[Tuple[str, List[str]]]:
    with open(path) as f:
        keyphrase_vocab = json.load(f)

    return keyphrase_vocab


def build_keyphrase_vocab_with_definition(
        keyphrase_vocab: List[Tuple[str, List[str]]],
        wiki_definition: Dict[str, str],
        num_keyphrases: int | None = None) -> List[Tuple[str, List[str], str]]:
    
    if num_keyphrases is not None:
        keyphrase_vocab = keyphrase_vocab[:num_keyphrases]

    keyphrase_vocab_with_definition = []
    num_with_definition = 0
    for canonical, members in keyphrase_vocab:
        for item in [canonical, *members]:
            definition = wiki_definition.get(item.lower())
            if definition: break

        if not definition or len(definition.split()) < 15:
            definition = canonical
        
        num_with_definition += 1

        keyphrase_vocab_with_definition.append((canonical, members, definition))

    print(f"{num_with_definition}/{len(keyphrase_vocab)} keyphrases have a wiki definition")
    return keyphrase_vocab_with_definition


def create_keyphrase_embedding(
        keyphrase_vocab_with_definition: List[Tuple[str, List[str], str]],
        batch_size: int,
        base_model: Dict[str, AutoModelForMaskedLM | AutoTokenizer],
        pooling: str = "mean") -> torch.Tensor:

    tokenizer = base_model["tokenizer"]
    norm = base_model["model"].model.get_input_embeddings().weight.norm(dim=1).mean().cpu()
    embeddings = []
    for i in tqdm(range(0, len(keyphrase_vocab_with_definition), batch_size), desc = "Creating embeddings"):
        _batch = keyphrase_vocab_with_definition[i:i+batch_size]

        if pooling == "mask":
            # Append a [MASK]; its bidirectional attention summarizes the definition.
            batch = [f"{tokenizer.mask_token}. Definition: {definition}" for _, _, definition in _batch]
        elif pooling == "mean_contextualized":
            # Concept is contextualized by its definition; we pool only the
            # concept tokens (which sit at the start of the template).
            concepts = [canonical for canonical, _, _ in _batch]
            batch = [f"{canonical} is {definition}" for canonical, _, definition in _batch]
        else:
            batch = [definition for _, _, definition in _batch]

        tok_kwargs = dict(return_tensors="pt", padding=True, truncation=True, max_length=1024)
        if pooling == "mean_contextualized":
            tok_kwargs.update(return_offsets_mapping=True, return_special_tokens_mask=True)
        inputs = tokenizer(batch, **tok_kwargs).to(base_model["model"].device)

        if pooling == "mean_contextualized":
            # Pop tensors the model's forward does not accept.
            offsets = inputs.pop("offset_mapping")            # [batch_size, seq, 2]
            special = inputs.pop("special_tokens_mask").bool()  # [batch_size, seq]

        with torch.no_grad():
            last_hidden = base_model["model"].model(**inputs).last_hidden_state  # [batch_size, seq, dim]

            if pooling == "mean":
                mask = inputs["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)  # [batch_size, seq, 1]
                hidden = (last_hidden * mask).sum(dim = 1) / mask.sum(dim = 1).clamp(min = 1e-8)  # [batch_size, dim]
            elif pooling == "mask":
                mask_positions = inputs["input_ids"] == tokenizer.mask_token_id  # [batch_size, seq]
                assert (mask_positions.sum(dim = 1) == 1).all(), \
                    "Each definition must contain exactly one [MASK] token."
                hidden = last_hidden[mask_positions]  # [batch_size, dim], row order preserved
            elif pooling == "mean_contextualized":
                concept_lens = torch.tensor(
                    [len(c) for c in concepts], device = last_hidden.device).unsqueeze(1)  # [batch_size, 1]
                starts = offsets[..., 0]  # [batch_size, seq]
                concept_mask = (~special) & (starts < concept_lens) & inputs["attention_mask"].bool()
                assert (concept_mask.sum(dim = 1) > 0).all(), \
                    "Found an example with no concept tokens (was the concept truncated?)."
                cm = concept_mask.unsqueeze(-1).to(last_hidden.dtype)  # [batch_size, seq, 1]
                hidden = (last_hidden * cm).sum(dim = 1) / cm.sum(dim = 1).clamp(min = 1e-8)  # [batch_size, dim]
            else:
                raise ValueError(
                    f"Unknown pooling '{pooling}'. Expected 'mean', 'mask', or 'mean_contextualized'.")

            hidden = base_model["model"].head(hidden.unsqueeze(0)).squeeze(0)
            hidden = hidden.cpu()
            hidden =  hidden * (norm / (hidden.norm(dim = 1, keepdim = True) + 1e-8))
            embeddings.append(hidden)

    return torch.cat(embeddings, dim = 0) #[num_keyphrases, dim]

def format_concept(keyphrase: str):
    words = keyphrase.split()
    return f"<<{' '.join(words)}>>"


def main():
    parser = ArgumentParser()
    parser.add_argument("--wiki-definition", type = str, required = True, help = "Path to wiki definition file")
    parser.add_argument("--keyphrase-vocab", type = str, required = True, help = "Path to keyphrase vocabulary")
    parser.add_argument("--input-model-name", type = str, default = "answerdotai/ModernBERT-base", help = "ModernBERT model from which we initiate our version")
    parser.add_argument("--output-model-path", type = str, required = True, help = "Where to save the output model")
    parser.add_argument("--keyphrase-vocab-size", type = int, default = None, help = "Number of keyphrases to add to our model")
    parser.add_argument("--batch-size", type = int, default = 32, help = "Batch size for embedding definitions")
    parser.add_argument("--pooling", type = str, default = "mean", choices = ["mean", "mask", "mean_contextualized"], help = "How to pool the definition into a keyphrase embedding")

    args = parser.parse_args()

    wiki_definition = load_wiki_definition(args.wiki_definition)
    keyphrase_vocab = load_keyphrase_vocab(args.keyphrase_vocab)

    keyphrase_vocab_with_definition = build_keyphrase_vocab_with_definition(
        keyphrase_vocab, wiki_definition, args.keyphrase_vocab_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_modernbert = {
        "model": AutoModelForMaskedLM.from_pretrained(args.input_model_name).to(device),
        "tokenizer": AutoTokenizer.from_pretrained(args.input_model_name)
    }

    original_vocab_size = base_modernbert["model"].config.vocab_size
    assert len(base_modernbert["tokenizer"]) == original_vocab_size, (
        f"Tokenizer length ({len(base_modernbert['tokenizer'])}) must equal "
        f"config.vocab_size ({original_vocab_size}) for phrase token ids to align "
        "with the phrase embedding partition."
    )
    num_keyphrases = len(keyphrase_vocab_with_definition)
    vocab_partitions = {
        "token": list(range(original_vocab_size)),
        "phrase": list(range(original_vocab_size, original_vocab_size + num_keyphrases)),
    }

    partitioned_modernbert = PartitionedModernBertForMaskedLM.from_pretrained(
        args.input_model_name, vocab_partitions = vocab_partitions)

    # Embed each keyphrase from its definition and write them into the phrase partition.
    keyphrase_embeddings = create_keyphrase_embedding(
        keyphrase_vocab_with_definition, args.batch_size, base_modernbert, pooling = args.pooling)

    partition_embeddings = {
        "phrase": keyphrase_embeddings,
    }
    partitioned_modernbert.add_embeddings(partition_embeddings)

    new_tokenizer = AutoTokenizer.from_pretrained(args.input_model_name)
    # add_tokens silently skips duplicates / pre-existing tokens. If it adds
    # fewer than num_keyphrases, the new token ids no longer line up with the
    # phrase embedding rows (which are never deduped), so fail loudly instead.
    num_added = new_tokenizer.add_tokens([
        format_concept(keyphrase) for keyphrase, _, _ in keyphrase_vocab_with_definition
    ])
    assert num_added == num_keyphrases, (
        f"add_tokens added {num_added} tokens but there are {num_keyphrases} "
        "keyphrases; duplicate or pre-existing formatted tokens would desync "
        "token ids from the phrase embedding partition."
    )

    os.makedirs(args.output_model_path, exist_ok = True)

    partitioned_modernbert.save_pretrained(args.output_model_path)
    new_tokenizer.save_pretrained(args.output_model_path)
    with open(os.path.join(args.output_model_path, "keyphrase_vocab.json"), "w") as f:
        json.dump(keyphrase_vocab_with_definition, f, indent = 2)

    print(f"Saved partitioned ModernBERT to {args.output_model_path}")


if __name__ == "__main__":
    main()
