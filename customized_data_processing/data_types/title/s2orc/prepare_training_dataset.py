# python prepare_training_dataset.py --input_file /scratch/lamdo/s2orc/processed/title_abstract_triplets/triplets_intermediate.tsv --metadata_file /scratch/lvnguyen/splade_keyphrases_expansion/dataset/s2orc_papers_metadata.jsonl --output_file /scratch/lamdo/s2orc/processed/title_abstract_triplets/raw.tsv
# python prepare_training_dataset.py --input_file /scratch/lamdo/s2orc/processed/title_abstract_triplets/triplets_intermediate.tsv --output_file /scratch/lamdo/s2orc/processed/title_abstract_triplets/raw_cs.tsv --metadata_file /scratch/lamdo/s2orc/processed/metadata_from_api/metadata_from_api.jsonl --fos_filter "Computer Science"

# python prepare_training_dataset.py --input_file /scratch/lamdo/s2orc/processed/title_abstract_triplets/triplets_intermediate_cs_fullsize.tsv --metadata_file /scratch/lamdo/s2orc/processed/metadata_from_api/metadata_from_api.jsonl --output_file /scratch/lamdo/s2orc/processed/title_abstract_triplets/raw_cs_fullsize.tsv
import json, os, random, string
from argparse import ArgumentParser
from tqdm import tqdm
from functools import lru_cache

PUNCTUATION = string.punctuation

@lru_cache(maxsize=1000000)
def process_title(title):
    return title.lower().strip().strip(PUNCTUATION)

def main():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type = str, required = True)
    parser.add_argument("--metadata_file", type = str, required = True)
    parser.add_argument("--output_file", type = str, required = True)
    parser.add_argument("--fos_filter", type = str, default = None, help = "Comma-separated list of Field of Studies to keep")

    args = parser.parse_args()

    input_file = args.input_file
    metadata_file = args.metadata_file
    output_file = args.output_file
    fos_filter = args.fos_filter

    fos_filter = [fos.strip() for fos in fos_filter.split(",")] if fos_filter else None

    triplets_intermediate = []
    with open(input_file) as f:
        for i, line in enumerate(tqdm(f, desc = "Reading triplets intermediate file")):
            splitted_line = line.split("\t")
            if len(splitted_line) != 3: continue
            triplets_intermediate.append(splitted_line)


    corpus_id_2_text = {}
    if fos_filter is not None: fos_filter_titles = set()
    else: fos_filter_titles = None
    with open(metadata_file) as f:
        error_count = 0
        for i, line in enumerate(tqdm(f, desc = "Reading metadata file")):
            try:
                jline = json.loads(line)
                corpus_id = jline.get("corpusId")
                title = jline.get("title")
                abstract = jline.get("abstract")

                if fos_filter is not None and any([fos in jline.get("fieldsOfStudy") for fos in fos_filter]):
                    fos_filter_titles.add(process_title(title))

                if not corpus_id or not title: continue

                corpus_id_2_text[corpus_id] = {
                    "title": title,
                    "abstract": abstract
                }
            except Exception:
                error_count += 1

        print("Number of errors in metadata", error_count)


    with open(output_file, "w") as f:
        for line in tqdm(triplets_intermediate):
            query, pos, neg_corpus_id = line
            if fos_filter_titles is not None and process_title(query) not in fos_filter_titles: continue

            if not query or not pos: continue

            try:
                neg_corpus_id = int(neg_corpus_id)
            except Exception: continue

            neg_metadata = corpus_id_2_text.get(neg_corpus_id)

            if not neg_metadata: continue

            neg_title = neg_metadata.get("title", "").strip(PUNCTUATION)
            neg_abstract = neg_metadata.get("abstract", "")
            neg = neg_title + ". " + neg_abstract if neg_title and neg_abstract else None
            if not neg: continue

            query = query.replace("\n", " ")
            pos = pos.replace("\n", " ")
            neg = neg.replace("\n", " ")

            f.write("\t".join([query, pos, neg]) + "\n")


if __name__ == "__main__":
    main()