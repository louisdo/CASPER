import random, os
from tqdm import tqdm

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

if __name__ == "__main__":

    files = {
        "erukgds": "/scratch/lamdo/unArxive/keyphrase_informativeness_combined_references/triplets_hardneg/raw.tsv",
        "kp": "/scratch/lamdo/phrase_splade_datasets/kp/raw.tsv",
        "kp1m": "/scratch/lamdo/phrase_splade_datasets/kp1m/raw.tsv",
        "cocit": "/scratch/lamdo/unArxive/cocit_training/raw.tsv",
        "fos": "/scratch/lamdo/phrase_splade_datasets/fos/raw.tsv",
        "mesh": "/scratch/lamdo/phrase_splade_datasets/mesh_descriptors/raw.tsv"
    }

    max_documents = {
        "erukgds": 1500000,
        # "kp": 500000,
        "cocit": 500000,
        "kp1m": 1000000
        # "fos": 250000,
        # "mesh": 250000
    }

    data_types_to_include = list(sorted(max_documents.keys()))
    data_types_to_include_str = "+".join(data_types_to_include)

    output_folder = f"/scratch/lamdo/phrase_splade_datasets/combined_{data_types_to_include_str}"
    output_file = os.path.join(output_folder, "raw.tsv")

    maybe_create_folder(output_folder)

    all_lines = []
    for data_type in data_types_to_include:
        file = files[data_type]
        with open(file) as infile:
            for i, line in enumerate(tqdm(infile, desc = f"Reading '{data_type}'")):
                if i == max_documents[data_type]: 
                    print(f"Stop. Need to read only {max_documents[data_type]} for data type {data_type}")
                    break
                all_lines.append(line)

    random.shuffle(all_lines)

    with open(output_file, "w") as f:
        for line in tqdm(all_lines, desc = "Writing data"):
            f.write(line)