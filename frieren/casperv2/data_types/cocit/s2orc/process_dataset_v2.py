# python process_dataset.py --input_folder /scratch/lamdo/s2orc/processed/extracted_metadata --output_file /scratch/lamdo/s2orc/processed/cocit_triplets/triplets_intermediate.tsv
# python process_dataset.py --input_folder "/scratch/lamdo/s2orc/processed/extracted_metadata_computer science" --output_file /scratch/lamdo/s2orc/processed/cocit_triplets/triplets_intermediate_cs_fullsize.tsv --max_samples_from_each_paper 1000000000

# python process_dataset.py --input_folder /scratch/lamdo/s2orc/processed/extracted_metadata --output_file /scratch/lamdo/s2orc/processed/cocit_triplets/triplets_intermediate_large.tsv --max_samples_from_each_paper 1000000000

import json, os, random, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from argparse import ArgumentParser
from tqdm import tqdm
from utils import get_context_from_metadata, load_metadata_from_api, write_tsv
from pyserini_lucence_bm25.build_index import PyseriniLuceneBM25


def get_context_from_metadata(metadata):

    if metadata["abstract"]:
        return " ".join([metadata["title"], metadata["abstract"]])
    return metadata["title"]


def get_negative_citation(
    group_citations, all_references, pos_venue, metadata_from_api_data, shared_venue
):

    # Filter references list based on venue
    if shared_venue:
        all_references = [
            reference
            for reference in all_references
            if reference in metadata_from_api_data
            and metadata_from_api_data[reference]["venue"] in pos_venue
        ]
    else:
        all_references = [
            reference
            for reference in all_references
            if reference in metadata_from_api_data
            and metadata_from_api_data[reference]["venue"] not in pos_venue
        ]

    if not all_references:
        return None

    for attempt in range(5):
        random_sampled_index = random.choice(range(len(all_references)))
        if all_references[random_sampled_index] in group_citations:
            continue

        return all_references[random_sampled_index]
    return None


def get_random_negative_from_corpus(citations, all_corpus_ids):
    random_sampled_index = None
    for attempt in range(3):
        random_sampled_index = random.choice(all_corpus_ids)
        if random_sampled_index not in citations:
            break

    return random_sampled_index


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--max_samples_from_each_paper", type=int, default=5)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--metadata_file", type=str, required=True)
    parser.add_argument("--special_token", type=str, required=True)

    args = parser.parse_args()

    input_folder = args.input_folder
    output_file = args.output_file
    max_samples_from_each_paper = args.max_samples_from_each_paper
    metadata_file = args.metadata_file
    special_token = args.special_token

    print("Loading files ...")
    files = os.listdir(input_folder)
    files = [
        os.path.join(input_folder, file) for file in files if file.endswith(".jsonl")
    ]
    print(files)

    print("Loading Corpus ...")
    all_corpus_ids, metadata_from_api_data = load_metadata_from_api(metadata_file)

    print("Loading Indexed BM25 ...")
    bm25_indexer = PyseriniLuceneBM25(
        index_path="/scratch/academic_online/s2orc/pyserini_index"
    )  # temp Osprey2
    bm25_indexer.load()

    cocit_triplets = []
    for file in tqdm(files, desc="Reading co-citations from files"):
        with open(file) as f:
            for line in tqdm(f):
                jline = json.loads(line)
                if not isinstance(jline, dict):
                    continue

                jline_cocit = jline.get("co_citation", [])
                all_references = jline.get("all_references", [])

                sampled_jline_cocit = random.sample(
                    jline_cocit, k=min(len(jline_cocit), max_samples_from_each_paper)
                )

                for cocit in sampled_jline_cocit:
                    cocit = [
                        idx
                        for idx in cocit
                        if idx in metadata_from_api_data
                        and metadata_from_api_data[idx]["abstract"]
                    ]
                    if len(cocit) < 2 or not all_references:
                        continue

                    pair = random.sample(cocit, k=2)
                    all_venues = [
                        metadata_from_api_data[pair[0]]["venue"],
                        metadata_from_api_data[pair[1]]["venue"],
                    ]

                    neg_citation_venue_idx = get_negative_citation(
                        cocit,
                        all_references,
                        all_venues,
                        metadata_from_api_data,
                        shared_venue=False,
                    )
                    neg_citation_concept_idx = get_negative_citation(
                        cocit,
                        all_references,
                        all_venues,
                        metadata_from_api_data,
                        shared_venue=True,
                    )

                    query = " ".join(metadata_from_api_data[pair[0]]["title"].split())
                    pos_doc = get_context_from_metadata(
                        metadata_from_api_data[pair[1]]
                    ).strip()

                    # Get tokens hard neg level and random context for dept level
                    query_similar_doc = bm25_indexer.search_by_bm25(
                        query, excluded_ids=pair, top_k=1
                    )
                    if not query_similar_doc:
                        continue

                    neg_citation_token_context = query_similar_doc[0]["content"]
                    neg_citation_dept_context = bm25_indexer.random_search(
                        excluded_ids=pair
                    )

                    if not neg_citation_token_context or not neg_citation_dept_context:
                        print("something went wrong")
                        continue

                    # Add logic so that in cases with no context (none context), the system uses a higher-level context instead.
                    neg_citation_venue_context = (
                        get_context_from_metadata(
                            metadata_from_api_data[neg_citation_venue_idx]
                        ).strip()
                        if neg_citation_venue_idx
                        else neg_citation_dept_context
                    )

                    neg_citation_concept_context = (
                        get_context_from_metadata(
                            metadata_from_api_data[neg_citation_concept_idx]
                        ).strip()
                        if neg_citation_concept_idx
                        else neg_citation_venue_context
                    )

                    all_neg_citation = [
                        neg_citation_dept_context,
                        neg_citation_venue_context,
                        neg_citation_concept_context,
                        neg_citation_token_context,
                    ]

                    cocit_triplets.append(
                        [query, pos_doc, special_token.join(all_neg_citation)]
                    )

    print(len(cocit_triplets))
    write_tsv(cocit_triplets, output_file)


if __name__ == "__main__":
    main()
