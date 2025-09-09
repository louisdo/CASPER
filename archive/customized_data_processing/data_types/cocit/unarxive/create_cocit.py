# python create_cocit.py --input_folder "/scratch/lamdo/unArxive" --output_file /scratch/lamdo/unArxive/cocitations.json

import json, os, random, re
from tqdm import tqdm
from argparse import ArgumentParser


def read_unarxive(folder = "/scratch/lamdo/unArxive"):
    subfolders = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", 
           "10", "11", "12", "13", "14", "15", "16", "17", "18", 
           "19", "20", "21", "22", "93", "97", "98"]
    
    subfolders = [os.path.join(folder, sf) for sf in subfolders]

    data = []
    for sf in tqdm(subfolders[:], desc = "Reading files"):
        files = os.listdir(sf)
        files_full_paths = [os.path.join(sf, file) for file in files]

        for file_full_path in tqdm(files_full_paths):
            with open(file_full_path) as f:
                for line in f:
                    data.append(json.loads(line))


    print("Number of papers in the dataset:", len(data))

    return data


def get_cocitations_from_paper(paper):
    bib_entries = paper["bib_entries"]
    body_text = paper["body_text"]


    all_groups = []
    for section in body_text:
        cite_spans = section["cite_spans"]
        if len(cite_spans) <= 1: continue

        section_groups = []
        current_group = [cite_spans[0]["ref_id"]]
        for i in range(1, len(cite_spans)):
            previous_end = cite_spans[i-1]["end"]
            current_start = cite_spans[i]["start"]

            if current_start - previous_end <= 5:
                current_group.append(cite_spans[i]["ref_id"])
            else:
                if len(current_group) >= 2: section_groups.append(current_group)
                current_group = []
        
        if len(current_group) >= 2: section_groups.append(current_group)

        if section_groups: all_groups.append(section_groups)

    all_groups_oa_id = []
    for section_groups in all_groups:
        section_groups_oa_id = []
        for group in section_groups:
            temp = []
            for ref_id in group:
                temp.append(bib_entries[ref_id].get("ids", {}).get("open_alex_id", ""))
            temp = [item for item in temp if item]
            if len(temp) > 1: section_groups_oa_id.append(temp)
        
        if section_groups_oa_id: all_groups_oa_id.extend(section_groups_oa_id)

    all_bib_entries_open_alex_ids = [bib_entries[ref_id].get("ids", {}).get("open_alex_id", "") for ref_id in bib_entries.keys()]
    all_bib_entries_open_alex_ids = [item for item in all_bib_entries_open_alex_ids if item]

    return {
        "cocit_groups": all_groups_oa_id,
        "citations": all_bib_entries_open_alex_ids
    }
        

def main():
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type = str, default = "/scratch/lamdo/unArxive")
    parser.add_argument("--output_file", type = str, required = True)

    args = parser.parse_args()

    input_folder = args.input_folder
    output_file = args.output_file

    papers = read_unarxive(input_folder)
    
    cocitations = []
    num_cocit_groups = 0
    for paper in tqdm(papers, desc = "Getting co-citations"):
        temp = get_cocitations_from_paper(paper)
        if temp["cocit_groups"]: 
            cocitations.append(temp)
            num_cocit_groups += len(temp["cocit_groups"])

    print("number of co citation groups", num_cocit_groups)

    with open(output_file, "w") as f:
        json.dump(cocitations, f, indent = 4)


if __name__ == "__main__":
    main()
