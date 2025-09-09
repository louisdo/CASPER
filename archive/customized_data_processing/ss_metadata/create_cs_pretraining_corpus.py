# python create_cs_pretraining_corpus.py --metadata_file /scratch/lamdo/s2orc/processed/metadata_from_api/metadata_from_api.jsonl --output_file /scratch/lamdo/s2orc/cs_corpus/collections.jsonl
import json
from argparse import ArgumentParser
from tqdm import tqdm

def main():
    parser = ArgumentParser()

    parser.add_argument("--metadata_file", type = str, required = True)
    parser.add_argument("--output_file", type = str, required = True)

    args = parser.parse_args()

    metadata_file = args.metadata_file
    output_file = args.output_file

    corpus = []
    with open(metadata_file) as f:
        error_count = 0
        for i, line in enumerate(tqdm(f, desc = "Reading metadata file")):
            try:
                jline = json.loads(line)
                corpus_id = jline.get("corpusId")
                title = jline.get("title")
                abstract = jline.get("abstract")

                if "Computer Science" in jline.get("fieldsOfStudy"):
                    to_append = {
                        "corpus_id": corpus_id,
                        "title": title,
                        "abstract": abstract
                    }
                    corpus.append(to_append)
            except Exception:
                error_count += 1

        print("Number of errors in metadata", error_count)

    with open(output_file, "w") as f:
        for line in tqdm(corpus, desc = "Writing corpus"):
            json.dump(line, f)
            f.write("\n")

if __name__ == "__main__":
    main()