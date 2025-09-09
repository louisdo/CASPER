import random, os
from tqdm import tqdm
from langdetect import detect as detect_language

def maybe_create_folder(folder_path):
    """
    Check if a folder exists and create it if it doesn't.

    Args:
    folder_path (str): The path of the folder to be created.

    Returns:
    bool: True if the folder was created, False if it already existed.
    """
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"Folder created: {folder_path}")
            return True
        except OSError as e:
            print(f"Error creating folder {folder_path}: {e}")
            return False
    else:
        print(f"Folder already exists: {folder_path}")
        return False


def check_document_high_quality(doc):
    # high quality here means lang is english and is long enough
    # if detect_language(doc) != "en":
    #     return False
    
    if len(doc.split(" ")) < 30:
        return False

    return True


if __name__ == "__main__":

    # files = {
    #     "erukgds": "/scratch/lamdo/unArxive/keyphrase_informativeness_combined_references/triplets_hardneg/raw.tsv",
    #     "kp": "/scratch/lamdo/phrase_splade_datasets/kp/raw.tsv",
    #     "kp1m": "/scratch/lamdo/phrase_splade_datasets/kp1m/raw.tsv",
    #     "cocit": "/scratch/lamdo/unArxive/cocit_training/raw.tsv",
    #     "fos": "/scratch/lamdo/phrase_splade_datasets/fos/raw.tsv",
    #     "mesh": "/scratch/lamdo/phrase_splade_datasets/mesh_descriptors/raw.tsv"
    # }

    # max_documents = {
    #     "erukgds": 1500000,
    #     # "kp": 500000,
    #     "cocit": 500000,
    #     "kp1m": 1000000
    #     # "fos": 250000,
    #     # "mesh": 250000
    # }

    files = {
        # "erukgds": "/scratch/lamdo/unArxive/keyphrase_informativeness_combined_references/triplets_hardneg/raw.tsv",
        # "kp": "/scratch/lamdo/phrase_splade_datasets/kp/raw.tsv",
        "kp1m": "/scratch/lamdo/phrase_splade_datasets/kp1m/raw.tsv",
        "kp20k": "/scratch/lamdo/phrase_splade_datasets/kp20k/raw.tsv",
        # "cocit": "/scratch/lamdo/unArxive/cocit_training/raw.tsv",
        # "fos": "/scratch/lamdo/phrase_splade_datasets/fos/raw.tsv",
        # "mesh": "/scratch/lamdo/phrase_splade_datasets/mesh_descriptors/raw.tsv"
        "cocit": "/scratch/lamdo/s2orc/processed/cocit_triplets/raw.tsv",
        "cocit_cs": "/scratch/lamdo/s2orc/processed/cocit_triplets/raw_cs.tsv",
        "cocit_cs_fullsize": "/scratch/lamdo/s2orc/processed/cocit_triplets/raw_cs_fullsize.tsv",
        "title": "/scratch/lamdo/s2orc/processed/title_abstract_triplets/raw.tsv", 
        "title_cs": "/scratch/lamdo/s2orc/processed/title_abstract_triplets/raw_cs.tsv", 
        "title_cs_fullsize": "/scratch/lamdo/s2orc/processed/title_abstract_triplets/raw_cs_fullsize.tsv", 
        "query": "/scratch/lamdo/s2orc/processed/query_triplets/raw.tsv",
        "query_cs": "/scratch/lamdo/s2orc/processed/query_triplets/raw_cs.tsv",
        "query_cs_fullsize": "/scratch/lamdo/s2orc/processed/query_triplets/raw_cs_fullsize.tsv",
        "cc": "/scratch/lamdo/s2orc/processed/citation_contexts_triplets/raw.tsv",
        "cc_cs": "/scratch/lamdo/s2orc/processed/citation_contexts_triplets/raw_cs.tsv",
        "cc_cs_fullsize": "/scratch/lamdo/s2orc/processed/citation_contexts_triplets/raw_cs_fullsize.tsv",

        "aol": "/scratch/lamdo/phrase_splade_datasets/aol_concept_annotations/raw.tsv"
    }
    max_documents = {
        # full set
        # "kp1m": 1500000,
        # "cocit": 1500000,
        # "title": 1500000,
        # "query": 1500000,
        # "cc": 1500000

        # domain-specific set: CS
        # "kp20k": 300000,
        # "cocit_cs": 300000,
        # "title_cs": 300000,
        # "query_cs": 300000,
        # "cc_cs": 300000,

        # ablation - no keyphrase
        # "cocit": 1500000,
        # "title": 1500000,
        # "query": 1500000,
        # "cc": 1500000,

        # ablation - no co-citation
        # "kp1m": 1500000,
        # "title": 1500000,
        # "query": 1500000,
        # "cc": 1500000,

        # ablation - no title
        # "kp1m": 1500000,
        # "cocit": 1500000,
        # "query": 1500000,
        # "cc": 1500000,

        # ablation - no query
        # "kp1m": 1500000,
        # "cocit": 1500000,
        # "title": 1500000,
        # "cc": 1500000,

        # ablation - no citation context
        # "kp1m": 1500000,
        # "cocit": 1500000,
        # "title": 1500000,
        # "query": 1500000,

        # for cs full size: APPLY_CHECK=False, for others, APPLY_CHECK=True
        # cs full size
        # "kp20k": 1500000,
        # "cocit_cs_fullsize": 1500000,
        # "title_cs_fullsize": 1500000,
        # "query_cs_fullsize": 1500000,
        # "cc_cs_fullsize": 1500000,



        # for aol APPLY_CHECK=False, for others, APPLY_CHECK=True
        # "cocit": 1500000,
        # "cc": 1500000,
        # "aol": 1500000
    }

    APPLY_CHECK = False


    data_types_to_include = list(sorted(max_documents.keys()))
    data_types_to_include_str = "+".join(data_types_to_include)

    output_folder = f"/scratch/lamdo/phrase_splade_datasets/combined_{data_types_to_include_str}"
    output_file = os.path.join(output_folder, "raw.tsv")

    maybe_create_folder(output_folder)

    all_lines = []
    for data_type in data_types_to_include:
        file = files[data_type]
        data_type_lines = []
        with open(file) as infile:
            for i, line in enumerate(tqdm(infile, desc = f"Reading '{data_type}'")):
                splitted_line = line.strip().split("\t")
                if len(splitted_line) != 3: continue

                query, pos, neg = splitted_line
                if APPLY_CHECK:
                    if check_document_high_quality(pos) and check_document_high_quality(neg): 
                        data_type_lines.append(line)
                else: data_type_lines.append(line)
        print(len(all_lines))

        sampled_data_type_lines = random.sample(data_type_lines, k = min(len(data_type_lines), max_documents[data_type]))
        all_lines.extend(sampled_data_type_lines)

    random.shuffle(all_lines)

    with open(output_file, "w") as f:
        for line in tqdm(all_lines, desc = "Writing data"):
            f.write(line)