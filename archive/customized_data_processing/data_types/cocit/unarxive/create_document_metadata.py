# python create_document_metadata.py --input_file /scratch/lamdo/unArxive/cocit_triplets.json --output_file /scratch/lamdo/unArxive/cocit_docs.json
import json, requests, time
from argparse import ArgumentParser
from tqdm import tqdm

def get_unique_openalex_id(triplets):
    res = set()
    for line in triplets:
        res.update(line)

    return res


def get_short_openalex_id(long_openalex_id):
    return long_openalex_id.split("/")[-1]

def inverted_index_to_string(inverted_index):
    if not inverted_index: return ""
    max_index = max(idx for indices in inverted_index.values() for idx in indices)

    words = [''] * (max_index + 1)

    for word, indices in inverted_index.items():
        for idx in indices:
            words[idx] = word
    
    return ' '.join(words)


def send_request_openalex(ids):
    url = "https://api.openalex.org/works"

    short_ids = [get_short_openalex_id(_id) for _id in ids]

    params = {
        "filter": "ids.openalex:" + "|".join(short_ids),
        "select": "id,title,abstract_inverted_index",
        "per-page": 200
    }

    response = requests.get(url, params=params)

    if response.status_code == 200: 
        resp_json = response.json()
        res = {}
        # print(resp_json["meta"]["count"])
        for line in resp_json["results"]:
            openalex_id = line["id"]
            line["abstract"] = inverted_index_to_string(line.get("abstract_inverted_index"))
            del line["abstract_inverted_index"]
            del line["id"]
            res[openalex_id] = line

        return res
    
    return {}




def get_document_data_from_openalex(openalex_ids, output_file):

    openalex_ids = list(openalex_ids)[:]
    
    with open(output_file, "w") as f:
        for i in tqdm(range(0, len(openalex_ids), 100)):
            batch = openalex_ids[i:i+100]
            resp = send_request_openalex(batch)

            json.dump(resp, f)
            f.write("\n")


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type = str, required = True)
    parser.add_argument("--output_file", type = str, required = True)

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    with open(input_file) as f:
        triplets = json.load(f)

    openalex_ids = get_unique_openalex_id(triplets)

    print(len(openalex_ids))

    get_document_data_from_openalex(openalex_ids, output_file)


if __name__ == "__main__":
    main()