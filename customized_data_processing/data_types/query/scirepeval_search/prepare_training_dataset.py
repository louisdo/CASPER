# python prepare_training_dataset.py --output_file /scratch/lamdo/s2orc/processed/query_triplets/raw.tsv
# python prepare_training_dataset.py --output_file /scratch/lamdo/s2orc/processed/query_triplets/raw_cs.tsv --metadata_file /scratch/lamdo/s2orc/processed/metadata_from_api/metadata_from_api.jsonl --fos_filter "Computer Science"

# python prepare_training_dataset.py --output_file /scratch/lamdo/s2orc/processed/query_triplets/raw_cs_fullsize.tsv --metadata_file /scratch/lamdo/s2orc/processed/metadata_from_api/metadata_from_api.jsonl --fos_filter "Computer Science"
import random, json
from datasets import load_dataset
from tqdm import tqdm
from argparse import ArgumentParser


def slightly_process_text(text):
    return text.replace("\n", "").replace("\t", "")


def write_tsv(triplets, output_file):
    with open(output_file, "w") as f:
        for line in tqdm(triplets, desc = "Writing dataset"):
            if len(line) != 3:
                print("Erroneous line!")
            to_write = "\t".join([str(item) for item in line])
            f.write(to_write + "\n")


def main():
    parser = ArgumentParser()
    parser.add_argument("--output_file", type = str, required = True)
    parser.add_argument("--metadata_file", type = str, default = None)
    parser.add_argument("--fos_filter", type = str, default = None)

    args = parser.parse_args()

    output_file = args.output_file
    metadata_file = args.metadata_file
    fos_filter = args.fos_filter

    fos_filter = [fos.strip() for fos in fos_filter.split(",")] if fos_filter else None

    if fos_filter is not None: 
        fos_filter_corpus_ids = set()
        assert metadata_file is not None
    else: fos_filter_corpus_ids = None
    with open(metadata_file) as f:
        error_count = 0
        for i, line in enumerate(tqdm(f, desc = "Reading metadata file")):
            try:
                jline = json.loads(line)
                corpus_id = int(jline.get("corpusId"))

                if fos_filter is not None and any([fos in jline.get("fieldsOfStudy") for fos in fos_filter]):
                    fos_filter_corpus_ids.add(corpus_id)
            except Exception:
                error_count += 1

        print("Number of errors in metadata", error_count)


    ds = load_dataset("allenai/scirepeval", "search")

    query_triplets = []
    for line in tqdm(ds["train"], desc = "Reading dataset and create triplets"):
        query = line.get("query")
        assert isinstance(query, str)
        query = slightly_process_text(query)

        positives = []
        negatives = []

        candidates = line.get("candidates")
        for cand in candidates:

            score = cand.get('score', 0)
            title = slightly_process_text(cand.get("title"))
            abstract = slightly_process_text(cand.get("abstract"))

            text = f"{title.lower()}. {abstract.lower()}"

            if score == 0:
                negatives.append(text)
            elif score > 0 and not (fos_filter and int(cand.get("doc_id")) not in fos_filter_corpus_ids): 
                positives.append(text)

        if not positives or not negatives: continue

        for pos in positives:
            if not pos: continue
            neg = random.choice(negatives)
            query_triplets.append([query, pos, neg])


    write_tsv(query_triplets, output_file)


if __name__ == "__main__":
    main()