import nltk, string
from nltk.stem.porter import PorterStemmer

STEMMER = PorterStemmer()

def stem_keyphrase(keyphrase: str):
    words = keyphrase.split(" ")
    words = [item for item in words if item]
    return " ".join([STEMMER.stem(w) for w in words])

def process_kp(_kps, remove_unigram = False):
    kps = _kps.copy()
    if remove_unigram == True:
        kps = [kp for kp in kps if len(kp.split(" ")) > 1]
    res = []
    for i in range(len(kps)):
        kp = kps[i].strip(string.punctuation).replace(" - ", " ").replace("-", " ")
        tokenized = nltk.word_tokenize(kp)
        processed_kp = []
        for tok in tokenized:
            processed_kp.append(STEMMER.stem(tok).lower())
        res.append(" ".join(processed_kp))
    return list(res)

def convert_list_of_text_to_list_of_tokens(text_list):
    res = set([])
    for item in text_list:
        res.update([tok for tok in item.split(" ") if tok])
    return list(res)

def precision_recall_f1(prediction: list, groundtruth: list, stem_prediction = True, stem_groundtruth = True):
    # convert to sets
    if not prediction or not groundtruth: return 0, 0 ,0
    if stem_prediction:
        set_prediction = set(process_kp(prediction))#set([stem_keyphrase(item) for item in prediction])
    else: 
        set_prediction = set(prediction)
    
    if stem_groundtruth:
        set_groundtruth = set(process_kp(groundtruth))#set([stem_keyphrase(item) for item in groundtruth])
    else: 
        set_groundtruth = set(groundtruth)

    prediction_groundtruth_intersection = set_groundtruth.intersection(set_prediction)

    if not prediction_groundtruth_intersection: return 0, 0, 0

    precision = len(prediction_groundtruth_intersection) / len(set_prediction)
    recall = len(prediction_groundtruth_intersection) / len(set_groundtruth)

    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def token_based_precision_recall_f1(prediction: list, groundtruth: list):
    prediction_token = convert_list_of_text_to_list_of_tokens(prediction)
    groundtruth_token = convert_list_of_text_to_list_of_tokens(groundtruth)

    return precision_recall_f1(prediction=prediction_token, groundtruth=groundtruth_token)