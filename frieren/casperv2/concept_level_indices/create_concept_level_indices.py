# python create_concept_level_indices.py --output_file /scratch/lamdo/casperv2/concept_level_indices/7Oct2025.json # with dep
# python create_concept_level_indices.py --output_file /scratch/lamdo/casperv2/concept_level_indices/17Oct2025.json # no dep

import json, os
import pandas as pd
from argparse import ArgumentParser
from transformers import AutoTokenizer
from collections import Counter


def main():
    parser = ArgumentParser()
    parser.add_argument("--casperv1_model_name", type = str, default = "lamdo/casper")
    parser.add_argument("--venue_names_data_path", type = str, default = "ggscholar_venues.csv")
    parser.add_argument("--dep_names_data_path", type = str, default = "all_depts.csv")
    parser.add_argument("--dep_level_min_frequency", type = int, default = 10)
    parser.add_argument("--venue_level_min_frequency", type = int, default = 1)
    parser.add_argument("--venue_min_hindex", type = int, default = 10)
    parser.add_argument("--output_file", type = str, required = True)


    args = parser.parse_args()

    # df_dep = pd.read_csv(args.dep_names_data_path)
    df_venue = pd.read_csv(args.venue_names_data_path)

    # all_dep_names = list(df_dep["department"])
    all_venue_names = set(df_venue[df_venue['h_index'] > args.venue_min_hindex]['full_name'])


    tokenizer = AutoTokenizer.from_pretrained(args.casperv1_model_name)
    reverse_voc = {v:k for k,v in tokenizer.vocab.items()}

    dep_phrase_counter = Counter()
    # for dep_name in all_dep_names:
    #     tokens = tokenizer(dep_name)["input_ids"]
    #     phrases = [tok for tok in tokens if tok >= 30522]

    #     dep_phrase_counter.update(phrases)
    
    dep_phrase_counter = {k:v for k,v in dep_phrase_counter.items() if v >= args.dep_level_min_frequency}
    
    venue_phrase_counter = Counter()
    for venue_name in all_venue_names:
        tokens = tokenizer(venue_name)["input_ids"]
        phrases = [tok for tok in tokens if tok >= 30522]

        venue_phrase_counter.update(phrases)

    venue_phrase_counter = {k:v for k,v in venue_phrase_counter.items() if k not in dep_phrase_counter and v >= args.venue_level_min_frequency}

    with open(args.output_file, "w") as f:
        json.dump({"dep": list(sorted(dep_phrase_counter.keys())), 
                   "venue": list(sorted(venue_phrase_counter.keys())),
                   "keyphrases": list([tok for tok in range(30522, len(reverse_voc)) if tok not in venue_phrase_counter and tok not in dep_phrase_counter]),
                   "tokens": list(range(30522))}, f)
    

if __name__ == "__main__":
    main()