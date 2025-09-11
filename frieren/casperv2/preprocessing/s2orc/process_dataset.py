import json, gzip
import os
import re
import requests
import wget
from tqdm import tqdm
from argparse import ArgumentParser
# from process_dataset_utils import extract_data_from_paper, maybe_create_folder


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


    if fos_filter is not None:
        raise NotImplementedError("Work in progress")
    

    