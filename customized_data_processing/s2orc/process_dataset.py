import json, gzip
import os
import re
import requests
import wget
from tqdm import tqdm
from argparse import ArgumentParser
from process_dataset_utils import extract_data_from_paper, maybe_create_folder

DATASET_NAME = "s2orc"

CORPUS_IDS_TO_KEEP = {
    "fos_filter_corpus_ids": None
}


def remove_file(file_path):
    """
    Removes the file at the specified path.

    Args:
        file_path (str): The path to the file to be removed.

    Returns:
        bool: True if the file was removed, False otherwise.
    """
    try:
        os.remove(file_path)
        print(f"File '{file_path}' has been removed.")
        return True
    except FileNotFoundError:
        print(f"File '{file_path}' does not exist.")
    except PermissionError:
        print(f"No permission to remove '{file_path}'.")
    except Exception as e:
        print(f"Error removing file '{file_path}': {e}")
    return False

def get_metadata_for_download(api_key):

    response = requests.get("https://api.semanticscholar.org/datasets/v1/release/latest").json()
    RELEASE_ID = response["release_id"]
    print(f"Latest release ID: {RELEASE_ID}")

    # get the download links for the s2orc dataset; needs to pass API key through `x-api-key` header
    # download via wget. this can take a while...
    response = requests.get(f"https://api.semanticscholar.org/datasets/v1/release/{RELEASE_ID}/dataset/{DATASET_NAME}/", headers={"x-api-key": api_key}).json()

    return response, RELEASE_ID



def download_shard(url, shard_idx, local_path, release_id):
    match = re.match(r"https://ai2-s2ag.s3.amazonaws.com/staging/(.*)/s2orc/(.*).gz(.*)", url)
    assert match.group(1) == release_id
    SHARD_ID = match.group(2)

    shard_path = os.path.join(local_path, f"{shard_idx}.gz")

    if not os.path.exists(shard_path):
        print(SHARD_ID)

        wget.download(url, out=shard_path)
    else: 
        print(f"File '{shard_path}' existed")

    return shard_path


def process_each_file(path):
    all_extracted_data = []
    with gzip.open(path,'rt') as f:
        for paper in tqdm(f):
            paper = json.loads(paper)

            if CORPUS_IDS_TO_KEEP["fos_filter_corpus_ids"] is not None and paper["corpusid"] not in CORPUS_IDS_TO_KEEP["fos_filter_corpus_ids"]:
                continue


            extracted_data = extract_data_from_paper(paper)
            all_extracted_data.append(extracted_data)

    return all_extracted_data


def main():
    parser = ArgumentParser()
    parser.add_argument("--api_key", type = str, required = True)
    parser.add_argument("--output_folder", type = str, required = True)
    parser.add_argument("--max_files", type = int, default = 10)
    parser.add_argument("--fos_filter", type = str, default = None)
    parser.add_argument("--metadata_file", type = str, default = None)

    args = parser.parse_args()

    api_key = args.api_key
    output_folder = args.output_folder
    max_files = args.max_files
    fos_filter = args.fos_filter
    metadata_file = args.metadata_file

    fos_filter = [fos.strip() for fos in fos_filter.split(",")] if fos_filter else None
    if fos_filter is not None:
        assert metadata_file is not None and os.path.exists(metadata_file)

        CORPUS_IDS_TO_KEEP["fos_filter_corpus_ids"] = set()

        with open(metadata_file) as f:
            error_count = 0
            for i, line in enumerate(tqdm(f, desc = "Reading metadata file")):
                try:
                    jline = json.loads(line)
                    corpus_id = int(jline.get("corpusId"))

                    if fos_filter is not None and any([fos in jline.get("fieldsOfStudy") for fos in fos_filter]):
                        CORPUS_IDS_TO_KEEP["fos_filter_corpus_ids"].add(corpus_id)
                except Exception:
                    error_count += 1

            print("Number of errors in metadata", error_count)


    if fos_filter is None:
        extracted_metadata_folder = os.path.join(output_folder, "extracted_metadata")
    else:
        fos_string = "+".join([fos.lower() for fos in fos_filter])
        extracted_metadata_folder = os.path.join(output_folder, f"extracted_metadata_{fos_string}")
    maybe_create_folder(extracted_metadata_folder)


    s2orc_temp_folder = os.path.join(output_folder, "s2orc_temp")
    maybe_create_folder(s2orc_temp_folder)

    download_metadata, release_id = get_metadata_for_download(api_key)

    for i, url in enumerate(tqdm(download_metadata["files"][:max_files])):
        shard_path = download_shard(url, i, s2orc_temp_folder, release_id)
        
        extracted_data_from_file = process_each_file(shard_path)
        outfile = os.path.join(extracted_metadata_folder, f"{i}.jsonl")

        with open(outfile, "w") as f:
            for line in extracted_data_from_file:
                json.dump(line, f)
                f.write("\n")

        # remove_file(shard_path)

    


if __name__ == "__main__":
    main()