# python process_dataset.py --input_folder /scratch/lamdo/s2orc/processed/extracted_metadata --output_file /scratch/lamdo/s2orc/processed/citation_contexts_triplets/triplets_intermediate.tsv
# python process_dataset.py --input_folder /scratch/lamdo/s2orc/processed/extracted_metadata --output_file /scratch/lamdo/s2orc/processed/citation_contexts_triplets/triplets_intermediate_cs.tsv --metadata_file /scratch/lamdo/s2orc/processed/metadata_from_api/metadata_from_api.jsonl --fos_filter "Computer Science"

# python process_dataset.py --input_folder /scratch/lamdo/s2orc/processed/extracted_metadata --output_file /scratch/lamdo/s2orc/processed/citation_contexts_triplets/triplets_intermediate_large.tsv --max_samples_from_each_paper 100000000

import json, os, random, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from argparse import ArgumentParser
from tqdm import tqdm
from utils import get_context_from_metadata, load_metadata_from_api, write_tsv
from pyserini_lucence_bm25.build_index import PyseriniLuceneBM25


def get_negative_citation(
    group_citations, all_references, pos_venue, metadata_from_api_data, shared_venue
):

    # Filter references list based on venue
    if shared_venue:
        all_references = [
            reference
            for reference in all_references
            if reference in metadata_from_api_data
            and pos_venue == metadata_from_api_data[reference]["venue"]
        ]
    else:
        all_references = [
            reference
            for reference in all_references
            if reference in metadata_from_api_data
            and pos_venue != metadata_from_api_data[reference]["venue"]
        ]

    if not all_references:
        return None

    for attempt in range(5):
        random_sampled_index = random.choice(range(len(all_references)))
        if all_references[random_sampled_index] in group_citations:
            continue

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
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--max_samples_from_each_paper", type=int, default=5)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--metadata_file", type=str, required=True)
    parser.add_argument("--special_token", type=str, required=True)

    args = parser.parse_args()

    input_folder = args.input_folder
    max_samples_from_each_paper = args.max_samples_from_each_paper
    output_file = args.output_file
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

    citation_context_triplets = []
    for file in tqdm(files, desc="Reading citation contexts from files"):
        with open(file) as f:
            for line in tqdm(f, desc=file):
                jline = json.loads(line)
                if not isinstance(jline, dict):
                    continue

                jline_cc = jline.get("citation_context", [])
                all_references = jline.get("all_references", [])

                if not all_references:
                    continue

                if isinstance(jline_cc, list):
                    #TODO: Comment the pipeline for better clearance 

                    sampled_jline_cc = random.sample(
                        jline_cc, k=min(len(jline_cc), max_samples_from_each_paper)
                    )

                    for cc in sampled_jline_cc:
                        # if not cc[1]: continue
                        cc[1] = [idx for idx in cc[1] if idx in all_corpus_ids]
                        if not cc[1]:
                            continue
                        text = cc[0]
                        citation = random.choice(cc[1])

                        # group_citations, all_references, pos_venue, metadata_from_api_data, shared_venue
                        neg_citation_venue_idx = get_negative_citation(
                            cc[1],
                            all_references,
                            metadata_from_api_data[citation]["venue"],
                            metadata_from_api_data,
                            shared_venue=False,
                        )
                        neg_citation_concept_idx = get_negative_citation(
                            cc[1],
                            all_references,
                            metadata_from_api_data[citation]["venue"],
                            metadata_from_api_data,
                            shared_venue=True,
                        )

                        # Get tokens hard neg level and random context for dept level
                        query_similar_doc = bm25_indexer.search_by_bm25(
                            text, excluded_ids=[citation], top_k=1
                        )
                        if not query_similar_doc: continue
                        neg_citation_token_context = query_similar_doc[0]['content']
                        neg_citation_dept_context = bm25_indexer.random_search(
                            excluded_ids=[citation]
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

                        citation_context_triplets.append(
                            [
                                text, #query 
                                get_context_from_metadata(
                                    metadata_from_api_data[citation]
                                ).strip(), #pos doc context
                                special_token.join(all_neg_citation), #group of 4 neg level
                            ]
                        )


    print(len(citation_context_triplets))
    write_tsv(citation_context_triplets, output_file)


if __name__ == "__main__":
    main()
