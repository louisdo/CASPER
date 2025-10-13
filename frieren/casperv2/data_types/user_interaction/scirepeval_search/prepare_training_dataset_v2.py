# python prepare_training_dataset.py --output_file /scratch/lamdo/s2orc/processed/query_triplets/raw.tsv
# python prepare_training_dataset.py --output_file /scratch/lamdo/s2orc/processed/query_triplets/raw_cs.tsv --metadata_file /scratch/lamdo/s2orc/processed/metadata_from_api/metadata_from_api.jsonl --fos_filter "Computer Science"

# python prepare_training_dataset.py --output_file /scratch/lamdo/s2orc/processed/query_triplets/raw_cs_fullsize.tsv --metadata_file /scratch/lamdo/s2orc/processed/metadata_from_api/metadata_from_api.jsonl --fos_filter "Computer Science"

import json, os, random, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import random, json
from datasets import load_dataset
from tqdm import tqdm
from argparse import ArgumentParser
from utils import write_tsv
from rapidfuzz import fuzz
from pyserini_lucence_bm25.build_index import PyseriniLuceneBM25


def slightly_process_text(text):
    return " ".join(text.split())


def get_context_from_metadata(metadata):
    if not metadata:
        return ""
    if type(metadata) != dict:
        return ""
    if metadata["abstract"]:
        content = " ".join([metadata["title"], metadata["abstract"]])
    else:
        content = metadata["title"]
    return slightly_process_text(content)


def main():
    parser = ArgumentParser()
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--metadata_file", type=str, default=None)
    parser.add_argument("--fos_filter", type=str, default=None)
    parser.add_argument("--special_token", type=str, required=True)

    args = parser.parse_args()

    output_file = args.output_file
    metadata_file = args.metadata_file
    fos_filter = args.fos_filter
    special_token = args.special_token

    fos_filter = [fos.strip() for fos in fos_filter.split(",")] if fos_filter else None

    if fos_filter is not None:
        fos_filter_corpus_ids = set()
        assert metadata_file is not None
    else:
        fos_filter_corpus_ids = None

    ds = load_dataset("allenai/scirepeval", "search")

    print("Loading Indexed BM25 ...")
    bm25_indexer = PyseriniLuceneBM25(
        index_path="/scratch/academic_online/s2orc/pyserini_index"
    )  # temp Osprey2
    bm25_indexer.load()

    query_triplets = []
    for line in tqdm(ds["train"], desc="Reading dataset and create triplets"):
        query = line.get("query")
        assert isinstance(query, str)

        query = slightly_process_text(query)

        candidates = line.get("candidates")
        positives = [cand for cand in candidates if cand.get("score", 0) == 1]
        negatives = [
            cand for cand in candidates if cand.get("score", 0) == 0
        ]  # and cand.get("venue") and cand.get("title") and cand.get("abstract")

        if not positives or not negatives:
            continue

        similar_docs_with_query = []
        for doc in bm25_indexer.search_by_bm25(
            query, excluded_ids=[], top_k=len(positives) * 2
        ):
            if any(
                [
                    fuzz.ratio(doc["content"], get_context_from_metadata(pos)) > 90
                    for pos in positives
                ]
            ):
                continue
            similar_docs_with_query.append(doc["content"])
            
        if not similar_docs_with_query:
            print("some problems when query hard neg candidates")
            continue


        for pos in positives:
            if not pos:
                continue
            if not pos.get("venue"):
                continue
            pos_content = get_context_from_metadata(pos)

            # choice for venue level
            venue_neg_candidates = [
                neg for neg in negatives if neg["venue"] != pos["venue"]
            ]
            venue_neg_metadata = (
                random.choice(venue_neg_candidates) if venue_neg_candidates else None
            )
            venue_neg_content = get_context_from_metadata(venue_neg_metadata)

            # choice for concept level
            concept_neg_candidates = [
                neg for neg in negatives if neg["venue"] != pos["venue"]
            ]
            concept_neg_metadata = (
                random.choice(concept_neg_candidates)
                if concept_neg_candidates
                else None
            )
            concept_neg_content = get_context_from_metadata(concept_neg_metadata)

            if not concept_neg_content and not venue_neg_content:
                continue

            # Get tokens hard neg level and random context for dept level
            neg_citation_token_context = random.choice(similar_docs_with_query)

            neg_citation_dept_context = bm25_indexer.random_search(excluded_ids=[])

            if not neg_citation_token_context or not neg_citation_dept_context:
                print("something went wrong")
                continue

            # Add logic so that in cases with no context (none context), the system uses a higher-level context instead.
            neg_citation_venue_context = (
                venue_neg_content if venue_neg_content else neg_citation_dept_context
            )

            neg_citation_concept_context = (
                concept_neg_content
                if concept_neg_content
                else neg_citation_venue_context
            )

            all_neg_citation = [
                neg_citation_dept_context,
                neg_citation_venue_context,
                neg_citation_concept_context,
                neg_citation_token_context,
            ]

            query_triplets.append(
                [
                    query,  # query
                    pos_content,  # pos doc context
                    special_token.join(all_neg_citation),  # group of 4 neg level
                ]
            )

    print(len(query_triplets))
    write_tsv(query_triplets, output_file)


if __name__ == "__main__":
    main()
