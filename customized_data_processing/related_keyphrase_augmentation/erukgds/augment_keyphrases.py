import json, os
from argparse import ArgumentParser
from tqdm import tqdm


def main():
    parser = ArgumentParser()

    parser.add_argument("--training_data_input_folder", type = str, required=True)
    parser.add_argument("--input_phrase_augmentation_file", type = str, required=True)
    parser.add_argument("--training_data_output_folder", type = str, required = True)
    parser.add_argument("--threshold", type = float, default = 0.2)
    parser.add_argument("--top_k_phrases", type = int, default = 5)

    args = parser.parse_args()

    training_data_input_folder = args.training_data_input_folder
    training_data_output_folder = args.training_data_output_folder
    input_phrase_augmentation_file = args.input_phrase_augmentation_file
    threshold = args.threshold
    top_k_phrases = args.top_k_phrases


    training_data_input_file = os.path.join(training_data_input_folder, "raw.tsv")
    training_data_output_file = os.path.join(training_data_output_folder, "raw.tsv")


    with open(input_phrase_augmentation_file) as f:
        queries_with_phrases = json.load(f)

    training_data = []
    with open(training_data_input_file) as f:
        for line in tqdm(f, desc = "Reading original training data"):
            splitted_line = line.split("\t")
            if len(splitted_line) != 3: continue
            training_data.append(splitted_line)


    with open(training_data_output_file, "w") as f:
        for line in tqdm(training_data, desc = "Augmenting phrases and write"):
            query, pos, neg = line

            phrases_for_query = queries_with_phrases.get(query, [])
            phrases_for_query = [item["phrase"] for item in phrases_for_query if item["score"] >= threshold][:top_k_phrases]

            phrases_for_query_string = ", ".join(phrases_for_query)

            new_query = f"{query}. {phrases_for_query_string}"

            to_write = "\t".join([new_query, pos, neg])
            f.write(to_write + "\n")


if __name__ == "__main__":
    main()