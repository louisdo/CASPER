# python create_triplets.py --input_file /scratch/lamdo/unArxive/cocitations.json --output_file /scratch/lamdo/unArxive/cocit_triplets.json

import json, itertools, random
from argparse import ArgumentParser
from tqdm import tqdm


def get_combinations(iterable, r):
    return list(itertools.combinations(iterable, r))


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type = str, required = True)
    parser.add_argument("--max_pairs_per_group", type = int, default = 8)
    parser.add_argument("--output_file", type=str, required = True)
    

    args = parser.parse_args()

    input_file = args.input_file
    max_pairs_per_group = args.max_pairs_per_group
    output_file = args.output_file

    with open(input_file) as f:
        cocitations = json.load(f)

    all_citations = set()
    for line in cocitations:
        all_citations.update(line["citations"])
    all_citations = list(all_citations)
    

    triplets = []
    for line in tqdm(cocitations[:]):
        cocit_groups = line["cocit_groups"]
        citations = set(line["citations"])

        if not citations or not cocit_groups:
            continue

        for group in cocit_groups:
            pairs = get_combinations(group, 2)

            negative_pool = list(citations - set(group))

            num_samples = min([len(pairs), max_pairs_per_group])

            sampled_pairs = random.sample(pairs, k = num_samples)

            if negative_pool:
                negative_samples = [random.choice(negative_pool) for _ in range(num_samples)]
            else:
                negative_samples = [random.choice(all_citations) for _ in range(num_samples)]

            for i in range(len(sampled_pairs)):
                query, pos = sampled_pairs[i]
                neg = negative_samples[i]

                triplets.append([query, pos, neg])

    with open(output_file, "w") as f:
        json.dump(triplets, f, indent = 4)


if __name__ == "__main__":
    main()