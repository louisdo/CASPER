import json, os, random
from argparse import ArgumentParser
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type = str, required = True)
    parser.add_argument("--max_samples_from_each_paper", type = int, default = 5)
    parser.add_argument("--output_file", type = str, required = True)


    args = parser.parse_args()

    input_folder = args.input_folder
    max_samples_from_each_paper = args.max_samples_from_each_paper
    output_file = args.output_file


    files = os.listdir(input_folder)
    files = [os.path.join(input_folder, file) for file in files if file.endswith(".jsonl")]