from pyserini_lucence_bm25.build_index import PyseriniLuceneBM25
from tqdm import tqdm
import json


def clean_content(content):
    return " ".join(content.split())


def get_context_from_metadata(metadata):
    if metadata["abstract"]:
        context = " ".join([metadata["title"], metadata["abstract"]])
    context = metadata["title"]
    return clean_content(context)


def load_metadata_from_api(metadata_file):
    print("Load_metadata_from_api ", metadata_file)
    
    metadata_from_api_data = {
        json.loads(line)["corpusId"]: json.loads(line)
        for line in tqdm(open(metadata_file).readlines())
        if "corpusId" in line 
        and type(json.loads(line)["abstract"]) == str
        and len(json.loads(line)["abstract"].split()) > 10
    }

    all_corpus_ids = list(metadata_from_api_data.keys())

    return all_corpus_ids, metadata_from_api_data


def write_tsv(triplets, output_file):
    with open(output_file, "w") as f:
        for line in tqdm(triplets, desc="Writing dataset"):
            if len(line) != 3:
                print("Erroneous line!")
            to_write = "\t".join([str(item) for item in line])
            f.write(to_write + "\n")


def test_query():

    # Create index instance
    indexer = PyseriniLuceneBM25(index_path="/scratch/academic_online/s2orc/pyserini_index")
    indexer.load()

    queries = [
        "CASPER: Concept-integrated Sparse Representation for Scientific Retrieval",
        "Attention Is All You Need",
        "A Comprehensive Overview of Large Language Models"
    ]*20

    for query in tqdm(queries):
        print("="*100)
        print(query)
        results = indexer.search_by_bm25(query, top_k=3)
        random_context = indexer.random_search()
        print(json.dumps(results, indent=4))
        print(random_context)
        print("\n")

if __name__ == "__main__":
    test_query()