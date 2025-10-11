# python process_dataset.py --input_folder /scratch/lamdo/s2orc/processed/extracted_metadata --output_file /scratch/lamdo/s2orc/processed/title_abstract_triplets/triplets_intermediate.tsv

# python process_dataset.py --input_folder "/scratch/lamdo/s2orc/processed/extracted_metadata_computer science" --output_file /scratch/lamdo/s2orc/processed/title_abstract_triplets/triplets_intermediate_cs_fullsize.tsv
import json, os, random, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from argparse import ArgumentParser
from tqdm import tqdm
from utils import get_context_from_metadata, load_metadata_from_api, write_tsv
from pyserini_lucence_bm25.build_index import PyseriniLuceneBM25
from rapidfuzz import fuzz


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type = str, required=True)
    parser.add_argument("--output_file", type = str, required = True)
    parser.add_argument("--metadata_file", type = str, required = True)
    parser.add_argument("--special_token", type = str, required = True)

    args = parser.parse_args()

    input_folder = args.input_folder
    output_file = args.output_file
    metadata_file = args.metadata_file
    special_token = args.special_token

    files = os.listdir(input_folder)
    files = [os.path.join(input_folder, file) for file in files if file.endswith(".jsonl")]

    print("Loading Indexed BM25 ...")
    bm25_indexer = PyseriniLuceneBM25(
        index_path="/scratch/academic_online/s2orc/pyserini_index"
    )  # temp Osprey2
    bm25_indexer.load()

    title_abstract_triplets = []
    for file in tqdm(files[:], desc = "Reading title-abstract from files"):
        with open(file) as f:
            for line in f:
                jline = json.loads(line)

                if not isinstance(jline, dict): continue

                jline_title_abstract = jline.get("title")
                all_references = jline.get("all_references")
                if not jline_title_abstract or not all_references: continue

                #query, doc
                title, abstract = jline_title_abstract

                #Get random context
                neg_citation_dept_context = bm25_indexer.random_search(
                    excluded_ids=[]
                )


                #Query Similar Doc for token level 
                similar_docs = []
                query_similar_doc = bm25_indexer.search_by_bm25(
                    title, excluded_ids=[], top_k=4
                )
                if not query_similar_doc: continue

                # Avoid the output is equal to this pos
                similar_docs = [
                    doc['content'] for doc in query_similar_doc
                    if fuzz.ratio(doc['content'], abstract) < 85
                ]
                if not similar_docs: continue


                neg = special_token.join([
                    neg_citation_dept_context, 
                    neg_citation_dept_context, 
                    neg_citation_dept_context, 
                    similar_docs[0]
                ])

                title_abstract_triplets.append([title, abstract, neg])

    write_tsv(title_abstract_triplets, output_file)



if __name__ == "__main__":
    main()