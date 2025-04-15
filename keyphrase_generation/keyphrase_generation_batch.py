import json, os
from argparse import ArgumentParser
from utils.nounphrase_extractor import CANDEXT, init_candext
from utils.splade_inference import SPLADE_MODEL, init_splade_model
from utils.process_dataset import process_dataset
from utils.keyphrase_generation_helper import keyphrase_generation_batch
from tqdm import tqdm



def main():
    parser = ArgumentParser()
    parser.add_argument("--splade_model_name", type = str, default = "phrase_splade")
    parser.add_argument("--dataset_name", type = str, default = "semeval")
    parser.add_argument("--output_folder", type = str, default = "/scratch/lamdo/phrase_splade_keyphrase_generation_results")

    args = parser.parse_args()

    splade_model_name = args.splade_model_name
    dataset_name = args.dataset_name
    output_folder = args.output_folder

    init_candext([1, 5])
    init_splade_model(splade_model_name)


    dataset = process_dataset(dataset_name)

    all_texts = [sample.get("text") for sample in dataset]

    generation_results = []
    BATCH_SIZE = 20
    for i in tqdm(range(0, len(all_texts), BATCH_SIZE), desc = "Extracting keyphrases"):
        batch = all_texts[i:i+BATCH_SIZE]

        to_extend = keyphrase_generation_batch(
            docs = batch,
            informativeness_model_name=splade_model_name,
            top_k=100,
            apply_position_penalty=True,
            length_penalty=-0.25,
            SPLADE_MODEL=SPLADE_MODEL,
            CANDEXT=CANDEXT["candext_1_5"]
        )
        generation_results.extend(to_extend)

    res = []
    for i, sample in tqdm(enumerate(dataset)):
        document = sample.get("text_not_lowered")
        present_keyphrases = sample.get("present_keyphrases")
        absent_keyphrases = sample.get("absent_keyphrases")

        automatically_extracted_keyphrases = generation_results[i]
        automatically_extracted_keyphrases = {
            "present_keyphrases":  [item[0] for item in automatically_extracted_keyphrases["present"] if item[1] > 0],
            "absent_keyphrases":  [item[0] for item in automatically_extracted_keyphrases["absent"] if item[1] > 0],
        }

        line = {
            "document": document,
            "present_keyphrases": present_keyphrases,
            "absent_keyphrases": absent_keyphrases,
            "automatically_extracted_keyphrases": automatically_extracted_keyphrases,
        }

        # if DATASET_TO_USE in RETRIEVAL_DATASETS:
        line.pop("document", None)


        res.append(line)

    output_file = os.path.join(output_folder, f"{dataset_name}--{splade_model_name}.json")
    with open(output_file, "w") as f:
        json.dump(res, f, indent = 4)



if __name__ == "__main__":
    main()