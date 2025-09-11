import json, glob, os, requests, gzip
from argparse import ArgumentParser
from tqdm import tqdm


def query_by_corpusIDs(corpusIDs, api_key): 
    try: 
        headers = {
            "x-api-key": api_key
        }

        r = requests.post(
            'https://api.semanticscholar.org/graph/v1/paper/batch',
            params={'fields': 'abstract,title,corpusId,fieldsOfStudy'},
            json={"ids": [f"CorpusID:{idx}" for idx in corpusIDs]}, 
            headers=headers 
        )

        results = r.json()

        return results
    except Exception as ve: 
        print(corpusIDs)
        return []

def flatten_list(nested): 
    return  [int(item) for sublist in nested for item in sublist]


def corpusids_from_scirepeval_search():
    from datasets import load_dataset

    ds = load_dataset("allenai/scirepeval", "search")
    
    corpusids = set()
    for line in tqdm(ds["train"], desc = "Reading dataset and create triplets"):
        candidates = line["candidates"]

        to_extend = [int(c["doc_id"]) for c in candidates]

        corpusids.update(to_extend)

    return corpusids

def get_existing_corpusids(output_file, from_scratch = False):
    # if output file already exists, get the corpus ids that we already
    # obtained metadata, so that we avoid retrieving metadata for them again

    if os.path.exists(output_file) and not from_scratch:
        res = set([])
        print("Found an existing metadata file, will avoid retrieving metadata for existing corpus ids")
        with open(output_file) as f:
            pbar = tqdm(f, desc = "Reading existing output file")
            error_count = 0
            for line in pbar:
                try:
                    jline = json.loads(line)
                    corpus_id = int(jline["corpusId"])

                    res.add(corpus_id)
                except Exception:
                    error_count += 1
                    pbar.set_postfix({"errors": error_count})
                    continue

            return res
    return set([])
        

def corpus_ids_from_s2orc_raw_folder(folder):
    res = set([])
    for file in tqdm(glob.glob(f"{folder}/*.gz")):
        with gzip.open(file,'rt') as f:
            for i, line in enumerate(f):
                jline = json.loads(line)
                corpus_id = jline["corpusid"]
                res.add(corpus_id)

    return res


def main():
    parser = ArgumentParser()
    parser.add_argument("--extracted_metadata_path", type = str, required = True)
    parser.add_argument("--semantic_scholar_api_key", type = str, required = True)
    parser.add_argument("--output_file", type = str, required = True)
    parser.add_argument("--get_corpusids_from_scirepeval_search", type = int, default=1, choices = [1,0])
    parser.add_argument("--s2orc_raw_folder", type = str, default = None)
    parser.add_argument("--from_scratch", type = int, default = 0, choices = [0, 1])

    args = parser.parse_args()


    extracted_metadata_path = args.extracted_metadata_path
    semantic_scholar_api_key = args.semantic_scholar_api_key
    output_file = args.output_file
    get_corpusids_from_scirepeval_search = args.get_corpusids_from_scirepeval_search
    s2orc_raw_folder = args.s2orc_raw_folder
    from_scratch = args.from_scratch

    existing_corpus_ids = get_existing_corpusids(output_file, from_scratch)

    all_ids_need_to_collect = set()
    if get_corpusids_from_scirepeval_search:
        all_ids_need_to_collect = corpusids_from_scirepeval_search()

    if s2orc_raw_folder:
        temp = corpus_ids_from_s2orc_raw_folder(s2orc_raw_folder)
        all_ids_need_to_collect = all_ids_need_to_collect.union(temp)


    for file in tqdm(glob.glob(f"{extracted_metadata_path}/*.jsonl")):
        data = [json.loads(line) for line in open(file)]
        for item in data: 
            if type(item) == list: continue
            all_ids_need_to_collect.update(flatten_list(item['co_citation']))
            all_ids_need_to_collect.update([int(docid) for docid in item['all_references']])

    if not from_scratch:
        print("Got", len(all_ids_need_to_collect))
        all_ids_need_to_collect = set([cid for cid in all_ids_need_to_collect if cid not in existing_corpus_ids])
        print("Only need to collect metadata for", len(all_ids_need_to_collect))
    else:
        print("Collecting metadata for", len(all_ids_need_to_collect), "corpus ids")

    all_ids_need_to_collect = list(sorted(all_ids_need_to_collect))

    for id in tqdm(range(0, len(all_ids_need_to_collect), 500)): 
        papers_info = query_by_corpusIDs(all_ids_need_to_collect[id:id+500], api_key = semantic_scholar_api_key)
        with open(output_file, 'a+' if not from_scratch else "w") as f: 
            f.write('\n'.join([json.dumps(line) for line in papers_info]) + '\n')

if __name__ == "__main__":
    main()