import numpy as np



###############################################################################
# stemming
###############################################################################
def stem_str_2d_list(str_2dlist, stemmer):
    # stem every word in a list of word list
    # str_list is a list of word list
    stemmed_str_2dlist = []
    for str_list in str_2dlist:
        stemmed_str_list = [stem_word_list(word_list, stemmer) for word_list in str_list]
        stemmed_str_2dlist.append(stemmed_str_list)
    return stemmed_str_2dlist

def stem_str_list(str_list, stemmer):
    # stem every word in a list of word list
    # str_list is a list of word list
    stemmed_str_list = []
    for word_list in str_list:
        stemmed_word_list = stem_word_list(word_list, stemmer)
        stemmed_str_list.append(stemmed_word_list)
    return stemmed_str_list

def stem_word_list(word_list, stemmer):
    return [stemmer.stem(w.strip().lower()) for w in word_list]


###############################################################################
# splitting pkp and akp
###############################################################################
def check_present_keyphrases(src_str, keyphrase_str_list, match_by_str=False):
    """
    :param src_str: stemmed word list of source text
    :param keyphrase_str_list: stemmed list of word list
    :return:
    """
    num_keyphrases = len(keyphrase_str_list)
    is_present = np.zeros(num_keyphrases, dtype=bool)

    for i, keyphrase_word_list in enumerate(keyphrase_str_list):
        joined_keyphrase_str = ' '.join(keyphrase_word_list)

        if joined_keyphrase_str.strip() == "":  # if the keyphrase is an empty string
            is_present[i] = False
        else:
            if not match_by_str:  # match by word
                # check if it appears in source text
                match = False
                for src_start_idx in range(len(src_str) - len(keyphrase_word_list) + 1):
                    match = True
                    for keyphrase_i, keyphrase_w in enumerate(keyphrase_word_list):
                        src_w = src_str[src_start_idx + keyphrase_i]
                        if src_w != keyphrase_w:
                            match = False
                            break
                    if match:
                        break
                if match:
                    is_present[i] = True
                else:
                    is_present[i] = False
            else:  # match by str
                if joined_keyphrase_str in ' '.join(src_str):
                    is_present[i] = True
                else:
                    is_present[i] = False
    return is_present

def separate_present_absent_by_source(src_token_list_stemmed, keyphrase_token_2dlist_stemmed, match_by_str):
    is_present_mask = check_present_keyphrases(src_token_list_stemmed, keyphrase_token_2dlist_stemmed, match_by_str)
    present_keyphrase_token2dlist = []
    absent_keyphrase_token2dlist = []
    for keyphrase_token_list, is_present in zip(keyphrase_token_2dlist_stemmed, is_present_mask):
        if is_present:
            present_keyphrase_token2dlist.append(keyphrase_token_list)
        else:
            absent_keyphrase_token2dlist.append(keyphrase_token_list)
    return present_keyphrase_token2dlist, absent_keyphrase_token2dlist, is_present_mask


###############################################################################
# filtering: checking unks and deduplify
###############################################################################
def check_duplicate_keyphrases(keyphrase_str_list):
    """
    :param keyphrase_str_list: a 2d list of tokens
    :return: a boolean np array indicate, 1 = unique, 0 = duplicate
    """
    num_keyphrases = len(keyphrase_str_list)
    not_duplicate = np.ones(num_keyphrases, dtype=bool)
    keyphrase_set = set()
    for i, keyphrase_word_list in enumerate(keyphrase_str_list):
        if '_'.join(keyphrase_word_list) in keyphrase_set:
            not_duplicate[i] = False
        else:
            not_duplicate[i] = True
        keyphrase_set.add('_'.join(keyphrase_word_list))
    return not_duplicate

def check_valid_keyphrases(str_list, UNK_WORD, invalidate_unk=True):
    num_pred_seq = len(str_list)
    is_valid = np.zeros(num_pred_seq, dtype=bool)
    for i, word_list in enumerate(str_list):
        keep_flag = True

        if len(word_list) == 0:
            keep_flag = False

        for w in word_list:
            if invalidate_unk:
                if w == UNK_WORD or w == ',' or w == '.':
                    keep_flag = False
            else:
                if w == ',' or w == '.':
                    keep_flag = False
        is_valid[i] = keep_flag

    return is_valid

def filter_prediction(disable_valid_filter, pred_token_2dlist_stemmed, UNK_TOKEN):
    """
    Remove the duplicate predictions, can optionally remove invalid predictions and extra one word predictions
    :param disable_valid_filter:
    :param pred_token_2dlist_stemmed:
    :param pred_token_2d_list:
    :return:
    """
    num_predictions = len(pred_token_2dlist_stemmed)
    is_unique_mask = check_duplicate_keyphrases(pred_token_2dlist_stemmed)  # boolean array, 1=unqiue, 0=duplicate
    pred_filter = is_unique_mask
    if not disable_valid_filter:
        is_valid_mask = check_valid_keyphrases(pred_token_2dlist_stemmed, UNK_TOKEN)
        pred_filter = pred_filter * is_valid_mask
    
    filtered_stemmed_pred_str_list = [word_list for word_list, is_keep in
                                      zip(pred_token_2dlist_stemmed, pred_filter) if
                                      is_keep]
    num_duplicated_predictions = num_predictions - np.sum(is_unique_mask)
    return filtered_stemmed_pred_str_list, num_duplicated_predictions, is_unique_mask

def find_unique_target(trg_token_2dlist_stemmed):
    """
    Remove the duplicate targets
    :param trg_token_2dlist_stemmed:
    :return:
    """
    num_trg = len(trg_token_2dlist_stemmed)
    is_unique_mask = check_duplicate_keyphrases(trg_token_2dlist_stemmed)  # boolean array, 1=unqiue, 0=duplicate
    trg_filter = is_unique_mask
    filtered_stemmed_trg_str_list = [word_list for word_list, is_keep in
                                     zip(trg_token_2dlist_stemmed, trg_filter) if
                                     is_keep]
    num_duplicated_trg = num_trg - np.sum(is_unique_mask)
    return filtered_stemmed_trg_str_list, num_duplicated_trg


###############################################################################
# averaging over the corpus (micro vs macro)
###############################################################################



###############################################################################
# other helpers
###############################################################################
def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())

def mae(a, b):
    return (np.abs(a - b)).mean()
