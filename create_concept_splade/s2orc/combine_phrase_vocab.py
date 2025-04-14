# python combine_phrase_vocab.py --input_folder ./vocab --output_file ./phrase_vocab.json

import json, os
from tqdm import tqdm
from argparse import ArgumentParser
from collections import Counter
from transformers import AutoTokenizer

def main():
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_file", type = str, required=True)
    parser.add_argument("--count_threshold", type = int, default = 100)
    parser.add_argument("--max_num_phrases", type = int, default = 20000)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert_vocab = tokenizer.vocab

    input_folder = args.input_folder
    output_file = args.output_file
    count_threshold = args.count_threshold
    max_num_phrases = args.max_num_phrases


    input_files = os.listdir(input_folder)
    input_files = [os.path.join(input_folder, file) for file in input_files]

    phrase_counter = Counter()
    for input_file in tqdm(input_files, desc = "Processing files"):
        with open(input_file) as f:
            data = json.load(f)
            phrase_counter.update(data)

    
    phrase_counter = Counter({k:v for k,v in phrase_counter.items() if v >= count_threshold and k not in bert_vocab})
    phrase_counter = phrase_counter.most_common(max_num_phrases)
    phrase_counter = {k:v for k,v in phrase_counter}

    print("Number of phrases", len(phrase_counter))

    with open(output_file, "w") as f:
        json.dump(phrase_counter, f, indent=4)


if __name__ == "__main__":
    main()