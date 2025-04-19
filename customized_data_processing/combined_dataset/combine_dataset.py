from tqdm import tqdm

if __name__ == "__main__":
    keyphrase_file = "/scratch/lamdo/splade_kp_datasets/kp20kbiomed/raw.tsv"
    erukgds_file = "/scratch/lamdo/unArxive/keyphrase_informativeness_combined_references/triplets_hardneg/raw.tsv"
    cocit_file = "/scratch/lamdo/unArxive/cocit_training/raw.tsv"

    output_file = "/scratch/lamdo/splade_kp_datasets/combined/raw.tsv"

    with open(output_file, "w") as outfile:
        for file in [keyphrase_file, erukgds_file, cocit_file]:
            with open(file) as infile:
                for line in tqdm(infile):
                    outfile.write(line + "\n")