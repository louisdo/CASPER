import json, gzip
import os
import re
import requests
import wget
from tqdm import tqdm
from argparse import ArgumentParser



def main():
    raise NotImplementedError("Work in progress")
    parser = ArgumentParser()
    parser.add_argument("--output_folder", type = str, required = True)
    parser.add_argument("--max_files", type = int, default = 10)

    args = parser.parse_args()
    output_folder = args.output_folder
    max_files = args.max_files


    s2orc_download_folder = os.path.join(output_folder, "s2orc_temp")