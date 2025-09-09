# python prepare_training_dataset.py --input_file /scratch/lamdo/s2orc/processed/citation_contexts_triplets/triplets_intermediate.tsv --metadata_file /scratch/lamdo/s2orc/processed/metadata_from_api/metadata_from_api.jsonl --output_file /scratch/lamdo/s2orc/processed/citation_contexts_triplets/raw.tsv
# python prepare_training_dataset.py --input_file /scratch/lamdo/s2orc/processed/citation_contexts_triplets/triplets_intermediate.tsv --metadata_file /scratch/lamdo/s2orc/processed/metadata_from_api/metadata_from_api.jsonl --output_file /scratch/lamdo/s2orc/processed/citation_contexts_triplets/raw_cs.tsv --fos_filter "Computer Science"

# python prepare_training_dataset.py --input_file /scratch/lamdo/s2orc/processed/citation_contexts_triplets/triplets_intermediate_large.tsv --metadata_file /scratch/lamdo/s2orc/processed/metadata_from_api/metadata_from_api.jsonl --output_file /scratch/lamdo/s2orc/processed/citation_contexts_triplets/raw_cs_fullsize.tsv --fos_filter "Computer Science" --require_abstract 0
import json, string
from argparse import ArgumentParser
from tqdm import tqdm

PUNCTUATION = string.punctuation

def main():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type = str, required = True)
    parser.add_argument("--metadata_file", type = str, required = True)
    parser.add_argument("--output_file", type = str, required = True)
    parser.add_argument("--fos_filter", type = str, default = None, help = "Comma-separated list of Field of Studies to keep")
    parser.add_argument("--require_abstract", type = int, default = 1)

    args = parser.parse_args()

    input_file = args.input_file
    metadata_file = args.metadata_file
    output_file = args.output_file
    fos_filter = args.fos_filter
    require_abstract = args.require_abstract
 
    fos_filter = [fos.strip() for fos in fos_filter.split(",")] if fos_filter else None

    triplets_intermediate = []
    with open(input_file) as f:
        for i, line in enumerate(tqdm(f, desc = "Reading triplets intermediate file")):
            splitted_line = line.split("\t")
            if len(splitted_line) != 3: continue
            try:
                query, pos_corpus_id, neg_corpus_id = splitted_line
                triplets_intermediate.append([query, int(pos_corpus_id), int(neg_corpus_id)])
            except Exception:
                continue


    corpus_id_2_text = {}
    if fos_filter is not None: fos_filter_corpus_ids = set()
    else: fos_filter_corpus_ids = None
    with open(metadata_file) as f:
        error_count = 0
        for i, line in enumerate(tqdm(f, desc = "Reading metadata file")):
            try:
                jline = json.loads(line)
                corpus_id = int(jline.get("corpusId"))
                title = jline.get("title")
                abstract = jline.get("abstract")

                if fos_filter is not None and any([fos in jline.get("fieldsOfStudy") for fos in fos_filter]):
                    fos_filter_corpus_ids.add(corpus_id)

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
            query, pos_corpus_id, neg_corpus_id = line
            
            if not query: continue
            if fos_filter_corpus_ids is not None and pos_corpus_id not in fos_filter_corpus_ids: continue

            pos_metadata = corpus_id_2_text.get(pos_corpus_id)
            neg_metadata = corpus_id_2_text.get(neg_corpus_id)

            if not pos_metadata or not neg_metadata: continue

            pos_title = pos_metadata.get("title", "").strip(PUNCTUATION)
            pos_title = pos_title if pos_title else ""
            pos_abstract = pos_metadata.get("abstract", "")
            pos_abstract = pos_abstract if pos_abstract else ""
            if require_abstract:
                pos = pos_title + ". " + pos_abstract if pos_title and pos_abstract else None
            else:
                pos = pos_title + ". " + pos_abstract if pos_title else None
            if not pos: continue


            neg_title = neg_metadata.get("title", "").strip(PUNCTUATION)
            neg_title = neg_title if neg_title else ""
            neg_abstract = neg_metadata.get("abstract", "")
            neg_abstract = neg_abstract if neg_abstract else ""
            if require_abstract:
                neg = neg_title + ". " + neg_abstract if neg_title and neg_abstract else None
            else:
                neg = neg_title + ". " + neg_abstract if neg_title else None
            if not neg: continue

            query = query.replace("\n", " ")
            pos = pos.replace("\n", " ")
            neg = neg.replace("\n", " ")

            f.write("\t".join([query, pos, neg]) + "\n")


if __name__ == "__main__":
    main()