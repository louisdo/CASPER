import os, random
from tqdm import tqdm
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type = str, required=True)
    parser.add_argument("--num_samples", type = int, required=True)
    parser.add_argument("--output_file", type = str, required = True)

    args = parser.parse_args()

    input_folder = args.input_folder
    num_samples = args.num_samples
    output_file = args.output_file


    input_file = os.path.join(input_folder, 'raw.tsv')

    data = []
    with open(input_file) as f:
        for line in f:
            data.append(line.strip("\n").strip())
    
    sampled_data = random.sample(data, k = min(len(data), num_samples))

    with open(output_file, "w") as f:
        for line in tqdm(sampled_data, desc = "Writing"):
            f.write(line)
            f.write("\n")

if __name__ == "__main__":
    main()