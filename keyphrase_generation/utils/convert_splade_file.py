import json 
import os 
from datasets import load_dataset

# def convert_file_(file, output = "_gitig_samples/"):
#     data = json.load(open(file)) 
#     present_keyphrases_data = []
#     absent_keyphrases_data = []

#     for idx, item in enumerate(data): 
#         present_keyphrases_data.append(
#             {
#                 "id": idx, 
#                 "source": " ; ".join(item['present_keyphrases'] + item['automatically_extracted_keyphrases']['present_keyphrases']), 
#                 "target": item['present_keyphrases'] + item['absent_keyphrases'], 
#                 "predictions": item['automatically_extracted_keyphrases']['present_keyphrases'] + item['automatically_extracted_keyphrases']['absent_keyphrases']
#             }
#         )

#         absent_keyphrases_data.append(
#             {
#                 "id": idx, 
#                 "source": " ; ".join(item['present_keyphrases'] + item['automatically_extracted_keyphrases']['present_keyphrases']), 
#                 "target": item['absent_keyphrases'], 
#                 "predictions": item['automatically_extracted_keyphrases']['absent_keyphrases']
#             }
#         )
    
#     with open(f"{output}_present_keyphrase_{file.split('/')[-1]}", 'w') as f:
#         f.write(json.dumps(present_keyphrases_data))
#     with open(f"{output}_absent_keyphrase_{file.split('/')[-1]}", 'w') as f:
#         f.write(json.dumps(absent_keyphrases_data))
#     return present_keyphrases_data, absent_keyphrases_data

def download_kp20k(): 
    import pandas as pd
    try: 
        dataset = pd.read_json("hf://datasets/memray/kp20k/test.json", lines=True)
    except: 
        dataset = load_dataset("memray/kp20k", split="test").to_pandas()
    
    all_titles = list(dataset['title'])
    all_abstracts = list(dataset['abstract'])

    sources = [f"{title} . {abstract}" for title, abstract in zip(all_titles, all_abstracts)]
    return sources

# def download_kp20k(): 
#     dataset = load_dataset("midas/kp20k", "generation", split="test")
#     return [" ".join(dataset[idx]['document']) for idx in range(len(dataset))]

def download_all_source(file): 
    file_name = file.split('/')[-1]
    dataset_type = file_name.split('--')[0]
    if dataset_type == "kp20k": 
        return download_kp20k()
    
    dataset = load_dataset(f"memray/{dataset_type}") 
    all_titles = list(dataset['test']['title'])
    all_abstracts = list(dataset['test']['abstract'])

    sources = [f"{title} . {abstract}" for title, abstract in zip(all_titles, all_abstracts)]

    return sources


def convert_file_(file, output = "_gitig_samples/", top_k = 10):
    data = json.load(open(file)) 
    sources = download_all_source(file)
    print(len(data), len(sources))
    assert len(data) == len(sources)

    all_keyphrases_data = []

    for idx, item in enumerate(data): 
        all_keyphrases_data.append(
            {
                "id": idx, 
                "source": sources[idx], 
                "target": item['present_keyphrases'] + item['absent_keyphrases'], 
                "predictions": item['automatically_extracted_keyphrases']['present_keyphrases'][:top_k] + item['automatically_extracted_keyphrases']['absent_keyphrases'][:top_k]
            }
        )

    
    with open(f"{output}_all_keyphrase_{file.split('/')[-1]}", 'w') as f:
        f.write(json.dumps(all_keyphrases_data, indent=4))



if __name__ == "__main__": 
    input_file = os.environ['input_file']
    output_dir = os.environ['output_dir']
    top_k = int(os.environ['top_k'])
    convert_file_(input_file, output_dir, top_k)
