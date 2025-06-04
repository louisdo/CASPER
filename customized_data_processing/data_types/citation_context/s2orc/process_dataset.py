# python process_dataset.py --input_folder /scratch/lamdo/s2orc/processed/extracted_metadata --output_file /scratch/lamdo/s2orc/processed/citation_contexts_triplets/triplets_intermediate.tsv
# python process_dataset.py --input_folder /scratch/lamdo/s2orc/processed/extracted_metadata --output_file /scratch/lamdo/s2orc/processed/citation_contexts_triplets/triplets_intermediate_cs.tsv --metadata_file /scratch/lamdo/s2orc/processed/metadata_from_api/metadata_from_api.jsonl --fos_filter "Computer Science"
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


def get_unique_corpus_id(citation_context_triplets):
    unique_corpus_id = set([])
    for line in citation_context_triplets:
        _, pos, neg = line
        unique_corpus_id.update([pos, neg])

    return unique_corpus_id


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

    citation_context_triplets = []
    for file in tqdm(files, desc = 'Reading citation contexts from files'):
        with open(file) as f:
            for line in f:
                jline = json.loads(line)
                if not isinstance(jline, dict): continue

                jline_cc = jline.get("citation_context", [])
                all_references = jline.get("all_references", [])

                if not all_references: continue

                if isinstance(jline_cc, list): 
                    sampled_jline_cc = random.sample(jline_cc, k = min(len(jline_cc), max_samples_from_each_paper))

                    for cc in sampled_jline_cc:
                        if not cc[1]: continue
                        text = cc[0]
                        citation = random.choice(cc[1])

                        neg_citation = get_negative_citation(cc[1], all_references)

                        if neg_citation: 
                            citation_context_triplets.append([text, citation, neg_citation])

    write_tsv(citation_context_triplets, output_file)

    

if __name__ == "__main__":
    main()