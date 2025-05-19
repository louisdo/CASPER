# python prepare_training_dataset.py --input_file /scratch/lamdo/s2orc/processed/title_abstract_triplets/triplets_intermediate.tsv --metadata_file /scratch/lvnguyen/splade_keyphrases_expansion/dataset/s2orc_papers_metadata.jsonl --output_file /scratch/lamdo/s2orc/processed/title_abstract_triplets/raw.tsv
import json, os, random, string
from argparse import ArgumentParser
from tqdm import tqdm


PUNCTUATION = string.punctuation


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type = str, required = True)
    parser.add_argument("--metadata_file", type = str, required = True)
    parser.add_argument("--output_file", type = str, required = True)

    args = parser.parse_args()

    input_file = args.input_file
    metadata_file = args.metadata_file
    output_file = args.output_file

    triplets_intermediate = []
    with open(input_file) as f:
        for i, line in enumerate(tqdm(f, desc = "Reading triplets intermediate file")):
            splitted_line = line.split("\t")
            if len(splitted_line) != 3: continue
            triplets_intermediate.append(splitted_line)


    corpus_id_2_text = {}
    with open(metadata_file) as f:
        error_count = 0
        for i, line in enumerate(tqdm(f, desc = "Reading metadata file")):
            try:
                jline = json.loads(line)
                corpus_id = jline.get("corpusId")
                title = jline.get("title")
                abstract = jline.get("abstract")

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