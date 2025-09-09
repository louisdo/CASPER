# CUDA_VISIBLE_DEVICES=2 python get_phrase_mask_embeddings.py --input_folder /scratch/lamdo/s2orc_phrase_vocab --max_numdocs_per_phrase 200 --output_file /scratch/lamdo/phrase_mask_embeddings.jsonl > error_logging.txt
import json, os, re, random, torch
from argparse import ArgumentParser
from datasets import load_dataset
from tqdm import tqdm
from utils import get_mask_embeddings, init_bert

def mask_random_occurrence(text:str, phrase:str, mask="[MASK]"):
    # Find all start indices of the phrase in the text
    matches = [m.start() for m in re.finditer(re.escape(phrase), text)]
    matches = [item for item in matches if item <= 1000]
    if not matches:
        return None  # Phrase not found, return original text

    # Choose one random occurrence
    idx = random.choice(matches)

    # Replace only the selected occurrence
    before = text[:idx]
    after = text[idx + len(phrase):]
    masked_text = before + mask + after
    return masked_text


def mask_phrase_in_texts(texts: list, phrase: str, mask:str="[MASK]"):
    res = [mask_random_occurrence(text, phrase, mask) for text in texts]
    res = [item for item in res if item]

    return res



def main():
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type = str, required = True)
    parser.add_argument("--mask", type = str, default = "[MASK]")
    parser.add_argument("--batch_size", type = int, default = 200)
    parser.add_argument("--bert_model_name", type = str, default = "distilbert-base-uncased")
    parser.add_argument("--output_file", type = str, default = "test_gitig_.json")
    parser.add_argument("--max_numdocs_per_phrase", type = int, default = -1)
    parser.add_argument("--count_threshold", type = int, default = 50)

    args = parser.parse_args()

    input_folder = args.input_folder
    mask = args.mask
    batch_size = args.batch_size
    bert_model_name = args.bert_model_name
    output_file = args.output_file
    max_numdocs_per_phrase = args.max_numdocs_per_phrase
    count_threshold = args.count_threshold

    files = os.listdir(input_folder)
    files = [os.path.join(input_folder, file) for file in files]

    init_bert(bert_model_name)

    ds = load_dataset("sentence-transformers/s2orc", "title-abstract-pair")

    phrase_occurrences = {}
    for file in tqdm(files, desc = "Reading files"):
        with open(file) as f:
            data = json.load(f)

        for k,v in data.items():
            if k not in phrase_occurrences: phrase_occurrences[k] = []

            phrase_occurrences[k].extend(v)

    # filter infrequent phrases
    phrase_occurrences = {k:v for k,v in phrase_occurrences.items() if len(v) >= count_threshold}
    print(len(phrase_occurrences))
    # random sample docs for each phrase
    phrase_occurrences = {k: random.sample(v, k = min(max_numdocs_per_phrase, len(v))) for k,v in phrase_occurrences.items()}

    all_doc_indices = set([])
    for v in phrase_occurrences.values():
        all_doc_indices.update(v)
    
    dataset = {}
    all_doc_indices = list(sorted(all_doc_indices))
    for index in tqdm(all_doc_indices, desc = "Load dataset into memory"):
        line = ds["train"][index]
        to_add = {
            "title": line["title"].lower().strip().replace("\n", " "),
            "abstract": line["abstract"].lower().strip().replace("\n", " ")
        }
        dataset[index] = to_add


    num_phrases = len(phrase_occurrences)

    sorted_phrases = list(sorted(phrase_occurrences.keys(), key = lambda x: -len(phrase_occurrences[x])))

    with open(output_file, "w") as f:

        pbar = tqdm(sorted_phrases, total = num_phrases)
        for phrase in pbar:
            try:
                doc_ids = phrase_occurrences[phrase]
                pbar.set_postfix({"num docs": len(doc_ids)})
                docs = []
                for doc_id in doc_ids:
                    title = dataset[doc_id]["title"]
                    abstract = dataset[doc_id]["abstract"]
                    text = f"{title}. {abstract}"
                    docs.append(text)


                masked_docs = mask_phrase_in_texts(texts = docs, phrase = phrase, mask = mask)

                accumulated_context_vector = None
                for i in range(0, len(masked_docs), batch_size):
                    masked_docs_batch = masked_docs[i:i+batch_size]
                    masked_embeddings_batch = get_mask_embeddings(texts = masked_docs_batch)

                    masked_embeddings_batch_sum_pooled = torch.sum(masked_embeddings_batch, dim = 0).cpu()
                    if accumulated_context_vector is None:
                        accumulated_context_vector = masked_embeddings_batch_sum_pooled
                    else: accumulated_context_vector += masked_embeddings_batch_sum_pooled

                if accumulated_context_vector is not None:
                    accumulated_context_vector /= len(masked_docs)
                    accumulated_context_vector = accumulated_context_vector.tolist()

                to_write = {"phrase": phrase, "emb": accumulated_context_vector}
                json.dump(to_write, f)
                f.write("\n")
            except Exception:
                print("Error:", phrase)

if __name__ == "__main__":
    main()
    