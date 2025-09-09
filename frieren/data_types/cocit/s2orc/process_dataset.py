# python process_dataset.py --input_folder /scratch/lamdo/s2orc/processed/extracted_metadata --output_file /scratch/lamdo/s2orc/processed/cocit_triplets/triplets_intermediate.tsv
# python process_dataset.py --input_folder "/scratch/lamdo/s2orc/processed/extracted_metadata_computer science" --output_file /scratch/lamdo/s2orc/processed/cocit_triplets/triplets_intermediate_cs_fullsize.tsv --max_samples_from_each_paper 1000000000

# python process_dataset.py --input_folder /scratch/lamdo/s2orc/processed/extracted_metadata --output_file /scratch/lamdo/s2orc/processed/cocit_triplets/triplets_intermediate_large.tsv --max_samples_from_each_paper 1000000000
import json, os, random
from argparse import ArgumentParser
from tqdm import tqdm


def write_tsv(triplets, output_file):
    with open(output_file, "w") as f:
        for line in tqdm(triplets, desc = "Writing dataset"):
            if len(line) != 3:
                print("Erroneous line!")
            to_write = "\t".join([str(item) for item in line])
            f.write(to_write + "\n")

def get_negative_citation(group_citations, all_references):
    for attempt in range(3):
        random_sampled_index = random.choice(range(len(all_references)))
        if all_references[random_sampled_index] in group_citations: continue

        return all_references[random_sampled_index]
    return None


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type = str, required = True)
    parser.add_argument("--max_samples_from_each_paper", type = int, default = 5)
    parser.add_argument("--output_file", type = str, required = True)

    args = parser.parse_args()

    input_folder = args.input_folder
    output_file = args.output_file
    max_samples_from_each_paper = args.max_samples_from_each_paper


    files = os.listdir(input_folder)
    files = [os.path.join(input_folder, file) for file in files if file.endswith(".jsonl")]

    cocit_triplets = []
    for file in tqdm(files[:], desc = "Reading co-citations from files"):
        with open(file) as f:
            for line in f:
                jline = json.loads(line)
                if not isinstance(jline, dict): continue

                jline_cocit = jline.get('co_citation', [])
                all_references = jline.get("all_references", [])

                sampled_jline_cocit = random.sample(jline_cocit, k = min(len(jline_cocit), max_samples_from_each_paper))

                for cocit in sampled_jline_cocit:
                    if len(cocit) < 2 or not all_references: continue
                    neg = get_negative_citation(cocit, all_references)
                    if not neg:
                        continue
                    
                    pair = random.sample(cocit, k = 2)
                    triplet = pair + [neg]

                    cocit_triplets.append(triplet)

    write_tsv(cocit_triplets, output_file)


if __name__ == "__main__":
    main()