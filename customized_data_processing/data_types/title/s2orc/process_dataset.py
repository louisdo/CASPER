# python process_dataset.py --input_folder /scratch/lamdo/s2orc/processed/extracted_metadata --output_file /scratch/lamdo/s2orc/processed/title_abstract_triplets/triplets_intermediate.tsv
import json, random, os
from argparse import ArgumentParser
from tqdm import tqdm


def write_tsv(triplets, output_file):
    with open(output_file, "w") as f:
        for line in tqdm(triplets, desc = "Writing dataset"):
            if len(line) != 3:
                print("Erroneous line!")
            to_write = "\t".join([str(item) for item in line])
            f.write(to_write + "\n")


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type = str, required=True)
    parser.add_argument("--output_file", type = str, required = True)

    args = parser.parse_args()

    input_folder = args.input_folder
    output_file = args.output_file

    files = os.listdir(input_folder)
    files = [os.path.join(input_folder, file) for file in files if file.endswith(".jsonl")]

    title_abstract_triplets = []
    for file in tqdm(files[:], desc = "Reading title-abstract from files"):
        with open(file) as f:
            for line in f:
                jline = json.loads(line)

                if not isinstance(jline, dict): continue

                jline_title_abstract = jline.get("title")
                all_references = jline.get("all_references")
                if not jline_title_abstract or not all_references: continue

                title, abstract = jline_title_abstract

                neg = random.choice(all_references)

                title_abstract_triplets.append([title, abstract, neg])

    write_tsv(title_abstract_triplets, output_file)



if __name__ == "__main__":
    main()