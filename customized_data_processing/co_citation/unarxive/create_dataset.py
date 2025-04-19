# python create_dataset.py --triplets_input_file /scratch/lamdo/unArxive/cocit_triplets.json --metadata_input_file /scratch/lamdo/unArxive/cocit_docs.json --output_file /scratch/lamdo/unArxive/cocit_training/raw.tsv
import json, string
from argparse import ArgumentParser
from tqdm import tqdm


def load_metadata(metadata_input_file):
    res = {}
    with open(metadata_input_file) as f:
        for line in tqdm(f, desc = "Loading document metadata"):
            jline = json.loads(line)
            res.update(jline)

    return res


def get_text_from_metadata(oa_id, metadata, only_title = False):
    doc_data = metadata.get(oa_id)
    if not doc_data: return ""

    title = doc_data.get("title").strip().strip(string.punctuation).replace("\n", " ").replace("\t", " ") if doc_data.get("title") else ""
    abstract = doc_data.get("abstract").strip().strip(string.punctuation).replace("\n", " ").replace("\t", " ") if doc_data.get("abstract") else ""

    if only_title:
        if not title: return ""
        return title

    if not title and not abstract: return ""
    elif title and abstract: return f"{title}. {abstract}"
    elif title and not abstract: return title
    else: return abstract

def main():
    parser = ArgumentParser()
    parser.add_argument("--triplets_input_file", type = str, required = True)
    parser.add_argument("--metadata_input_file", type = str, required = True)
    parser.add_argument("--output_file", type = str, required = True)

    args = parser.parse_args()

    triplets_input_file = args.triplets_input_file
    metadata_input_file = args.metadata_input_file
    output_file = args.output_file

    assert output_file.endswith(".tsv"), "Output must be a tsv file"


    metadata = load_metadata(metadata_input_file)

    with open(triplets_input_file) as f:
        triplets = json.load(f)


    with open(output_file, "w") as f:
        count = 0
        for triplet in tqdm(triplets, desc = "Processing and writing output"):
            query_oa_id, pos_oa_id, neg_oa_id = triplet

            query = get_text_from_metadata(query_oa_id, metadata, only_title=True)
            pos = get_text_from_metadata(pos_oa_id, metadata)
            neg = get_text_from_metadata(neg_oa_id, metadata)

            if not query or not pos or not neg: continue

            to_write = "\t".join([query, pos, neg])

            f.write(to_write)
            f.write("\n")

            count += 1


    print("Number of training samples created", count)

        


if __name__ == "__main__":
    main()