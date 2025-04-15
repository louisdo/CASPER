import os, json
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from evaluation_metrics.f_score import precision_recall_f1, token_based_precision_recall_f1, stem_keyphrase, process_kp
from nltk.stem.porter import PorterStemmer

STEMMER = PorterStemmer()


KEYPHRASE_STEMMING_REQUIREMENTS = {
    "semeval": {"prediction": False, "groundtruth": True},
    "inspec": {"prediction": False, "groundtruth": True},
    "nus": {"prediction": False, "groundtruth": True},
    "krapivin": {"prediction": False, "groundtruth": True},
    "kp20k": {"prediction": False, "groundtruth": True},
}


def remove_duplicates(stemmed_keyphrases):
    visited = set([])
    indices = []
    for index, item in enumerate(stemmed_keyphrases):
        _temp = item
        if _temp in visited: continue
        indices.append(index)
        visited.add(_temp)
    
    return [stemmed_keyphrases[index] for index in indices]


def format_datetime_to_string(dt = datetime.now(), format_str="%Y-%m-%d %H:%M:%S"):
    """
    Formats a datetime object to a string using the provided format string.
    
    Args:
        dt (datetime): The datetime object to format.
        format_str (str): The format string to use (default is "%Y-%m-%d %H:%M:%S").
        
    Returns:
        str: The formatted datetime string.
    """
    return dt.strftime(format_str)

def macro_precision_recall_f1(predictions, groundtruths, top_k, stem_prediction = True, stem_groundtruth = True):
    assert len(predictions) == len(groundtruths)
    relevant_indices = [i for i in range(len(groundtruths)) if groundtruths[i]]

    print(len(predictions), len(relevant_indices), top_k)

    predictions = [predictions[i] for i in relevant_indices]
    groundtruths = [groundtruths[i] for i in relevant_indices]

    prfs = [precision_recall_f1(prediction = prediction[:top_k], 
                                    groundtruth=groundtruth,
                                    stem_prediction = stem_prediction,
                                    stem_groundtruth=stem_groundtruth)\
                                          for prediction, groundtruth in tqdm(zip(predictions, groundtruths), desc = f"Eval P, R, F for top {top_k}")]
        
    # assert len(prfs) == len(result_data)
    f1s = [item[2] for item in prfs]
    precisions = [item[0] for item in prfs]
    recalls = [item[1] for item in prfs]
    f1_macro = np.mean(f1s)
    precision_macro = np.mean(precisions)
    recall_macro = np.mean(recalls)

    return precision_macro, recall_macro, f1_macro


if __name__ == "__main__":
    RESULTS_FOLDER = os.environ["RESULTS_FOLDER"]
    DATASETS_TO_INCLUDE = os.getenv("DATASETS_TO_INCLUDE") # if this is specified, only evaluate the performance on those datasets
    MODELS_TO_INCLUDE = os.getenv("MODELS_TO_INCLUDE") # if this is specified, only evaluate the performance on those models

    if DATASETS_TO_INCLUDE:
        DATASETS_TO_INCLUDE = [item.strip() for item in DATASETS_TO_INCLUDE.split(",")]
    files = os.listdir(RESULTS_FOLDER)

    if MODELS_TO_INCLUDE:
        MODELS_TO_INCLUDE = [item.strip() for item in MODELS_TO_INCLUDE.split(",")]
    print(MODELS_TO_INCLUDE)
    
    eval_lines = []
    for file in files:
        dataset_name, model_name = file.replace(".json", "").split("--")

        if DATASETS_TO_INCLUDE and dataset_name not in DATASETS_TO_INCLUDE:
            continue
        if MODELS_TO_INCLUDE and model_name not in MODELS_TO_INCLUDE:
            continue

        dataset_stemming_requirements = KEYPHRASE_STEMMING_REQUIREMENTS.get(dataset_name, {})

        with open(os.path.join(RESULTS_FOLDER, file)) as f:
            result_data = json.load(f)

        eval_line = {
            "dataset_name": dataset_name,
            "model_name": model_name,
        }

        all_predicted_present_keyphrases = [line.get("automatically_extracted_keyphrases", {}).get("present_keyphrases", []) for line in result_data]
        # stem the predicted keyphrases
        all_predicted_present_keyphrases = [process_kp(present_keyphrases_line) for present_keyphrases_line in all_predicted_present_keyphrases]
        # remove predicted keyphrases after stemming
        all_predicted_present_keyphrases = [remove_duplicates(stemmed_keyphrases) + [f"<padding-{padding_index}>" for padding_index in range(10)] for stemmed_keyphrases in all_predicted_present_keyphrases]

        # do the same for absent keyphrases
        all_predicted_absent_keyphrases = [line.get("automatically_extracted_keyphrases", {}).get("absent_keyphrases", [])+ [f"<padding-{padding_index}>" for padding_index in range(10)]  for line in result_data]
        all_predicted_absent_keyphrases = [process_kp(absent_keyphrases_line) for absent_keyphrases_line in all_predicted_absent_keyphrases]
        all_predicted_absent_keyphrases = [remove_duplicates(stemmed_keyphrases) + [f"<padding-{padding_index}>" for padding_index in range(10)] for stemmed_keyphrases in all_predicted_absent_keyphrases]

        present_keyphrases_groundtruths = [line.get("present_keyphrases") for line in result_data]
        absent_keyphrases_groundtruths = [line.get("absent_keyphrases") for line in result_data]

        for top_k in [3,5,10]:
            # present keyphrases
            present_keyphrases_predictions = all_predicted_present_keyphrases

            _, _, eval_line[f"present-f1@{top_k}"] = macro_precision_recall_f1(predictions = present_keyphrases_predictions, 
                                                                        groundtruths=present_keyphrases_groundtruths,
                                                                        top_k = top_k,
                                                                        stem_prediction = dataset_stemming_requirements.get("prediction"),
                                                                        stem_groundtruth = dataset_stemming_requirements.get("groundtruth"))
            
            # absent keyphrases
            absent_keyphrases_predictions = all_predicted_absent_keyphrases
            _, eval_line[f"absent-recall@{top_k}"], _ = macro_precision_recall_f1(predictions = absent_keyphrases_predictions, 
                                                                            groundtruths=absent_keyphrases_groundtruths,
                                                                            top_k = top_k,
                                                                            stem_prediction = dataset_stemming_requirements.get("prediction"),
                                                                        stem_groundtruth = dataset_stemming_requirements.get("groundtruth"))

        eval_lines.append(eval_line)

    eval_lines = list(sorted(eval_lines, key = lambda x: [x.get("dataset_name"), x.get("model_name")]))


    df = pd.DataFrame(eval_lines)
    df.to_csv("view.csv", index = False)