import gin
import numpy as np
from tqdm import tqdm

from metrics import KeyphraseMetric
from metrics.metric_utils import separate_present_absent_by_source, filter_prediction, find_unique_target

from collections import defaultdict


def update_score_dict(trg_token_2dlist_stemmed, pred_token_2dlist_stemmed, k_list, score_dict, tag):
    num_targets = len(trg_token_2dlist_stemmed)
    num_predictions = len(pred_token_2dlist_stemmed)

    is_match = compute_match_result(trg_token_2dlist_stemmed, pred_token_2dlist_stemmed,
                                    type='exact', dimension=1)
    is_match_substring_2d = compute_match_result(trg_token_2dlist_stemmed,
                                                 pred_token_2dlist_stemmed, type='sub', dimension=2)
    
    # Classification metrics
    precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks = \
        compute_classification_metrics_at_ks(is_match, num_predictions, num_targets, k_list=k_list,
                                             meng_rui_precision=False)

    # Ranking metrics
    ndcg_ks, dcg_ks = ndcg_at_ks(is_match, k_list=k_list, num_trgs=num_targets, method=1, include_dcg=True)
    alpha_ndcg_ks, alpha_dcg_ks = alpha_ndcg_at_ks(is_match_substring_2d, k_list=k_list, method=1,
                                                   alpha=0.5, include_dcg=True)
    ap_ks = average_precision_at_ks(is_match, k_list=k_list,
                                    num_predictions=num_predictions, num_trgs=num_targets)

    #for topk, precision_k, recall_k, f1_k, num_matches_k, num_predictions_k in \
    #        zip(k_list, precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks):
    for topk, precision_k, recall_k, f1_k, num_matches_k, num_predictions_k, ndcg_k, dcg_k, alpha_ndcg_k, alpha_dcg_k, ap_k in \
        zip(k_list, precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks, ndcg_ks, dcg_ks,
            alpha_ndcg_ks, alpha_dcg_ks, ap_ks):
        score_dict['precision@{}_{}'.format(topk, tag)].append(precision_k)
        score_dict['recall@{}_{}'.format(topk, tag)].append(recall_k)
        score_dict['f1_score@{}_{}'.format(topk, tag)].append(f1_k)
        score_dict['num_matches@{}_{}'.format(topk, tag)].append(num_matches_k)
        score_dict['num_predictions@{}_{}'.format(topk, tag)].append(num_predictions_k)
        score_dict['num_targets@{}_{}'.format(topk, tag)].append(num_targets)
        score_dict['AP@{}_{}'.format(topk, tag)].append(ap_k)
        score_dict['NDCG@{}_{}'.format(topk, tag)].append(ndcg_k)
        score_dict['AlphaNDCG@{}_{}'.format(topk, tag)].append(alpha_ndcg_k)

    score_dict['num_targets_{}'.format(tag)].append(num_targets)
    score_dict['num_predictions_{}'.format(tag)].append(num_predictions)
    return score_dict


def compute_match_result(trg_str_list, pred_str_list, type='exact', dimension=1):
    assert type in ['exact', 'sub'], "Right now only support exact matching and substring matching"
    assert dimension in [1, 2], "only support 1 or 2"
    num_pred_str = len(pred_str_list)
    num_trg_str = len(trg_str_list)
    if dimension == 1:
        is_match = np.zeros(num_pred_str, dtype=bool)
        for pred_idx, pred_word_list in enumerate(pred_str_list):
            joined_pred_word_list = ' '.join(pred_word_list)
            for trg_idx, trg_word_list in enumerate(trg_str_list):
                joined_trg_word_list = ' '.join(trg_word_list)
                if type == 'exact':
                    if joined_pred_word_list == joined_trg_word_list:
                        is_match[pred_idx] = True
                        break
                elif type == 'sub':
                    if joined_pred_word_list in joined_trg_word_list:
                        is_match[pred_idx] = True
                        break
    else:
        is_match = np.zeros((num_trg_str, num_pred_str), dtype=bool)
        for trg_idx, trg_word_list in enumerate(trg_str_list):
            joined_trg_word_list = ' '.join(trg_word_list)
            for pred_idx, pred_word_list in enumerate(pred_str_list):
                joined_pred_word_list = ' '.join(pred_word_list)
                if type == 'exact':
                    if joined_pred_word_list == joined_trg_word_list:
                        is_match[trg_idx][pred_idx] = True
                elif type == 'sub':
                    if joined_pred_word_list in joined_trg_word_list:
                        is_match[trg_idx][pred_idx] = True
    return is_match


def compute_classification_metrics_at_ks(
        is_match, num_predictions, num_trgs, k_list=[5, 10], meng_rui_precision=False
):
    """
    :param is_match: a boolean np array with size [num_predictions]
    :param predicted_list:
    :param true_list:
    :param topk:
    :return: {'precision@%d' % topk: precision_k, 'recall@%d' % topk: recall_k, 'f1_score@%d' % topk: f1, 'num_matches@%d': num_matches}
    """
    assert is_match.shape[0] == num_predictions
    # topk.sort()
    if num_predictions == 0:
        precision_ks = [0] * len(k_list)
        recall_ks = [0] * len(k_list)
        f1_ks = [0] * len(k_list)
        num_matches_ks = [0] * len(k_list)
        num_predictions_ks = [0] * len(k_list)
    else:
        num_matches = np.cumsum(is_match)
        num_predictions_ks = []
        num_matches_ks = []
        precision_ks = []
        recall_ks = []
        f1_ks = []
        for topk in k_list:
            if topk == 'M':
                topk = num_predictions
            elif topk == 'G':
                # topk = num_trgs
                if num_predictions < num_trgs:
                    topk = num_trgs
                else:
                    topk = num_predictions
            elif topk == 'O':
                topk = num_trgs

            if meng_rui_precision:
                if num_predictions > topk:
                    num_matches_at_k = num_matches[topk - 1]
                    num_predictions_at_k = topk
                else:
                    num_matches_at_k = num_matches[-1]
                    num_predictions_at_k = num_predictions
            else:
                if num_predictions > topk:
                    num_matches_at_k = num_matches[topk - 1]
                else:
                    num_matches_at_k = num_matches[-1]
                num_predictions_at_k = topk

            precision_k, recall_k, f1_k = compute_classification_metrics(
                num_matches_at_k, num_predictions_at_k, num_trgs
            )
            precision_ks.append(precision_k)
            recall_ks.append(recall_k)
            f1_ks.append(f1_k)
            num_matches_ks.append(num_matches_at_k)
            num_predictions_ks.append(num_predictions_at_k)
    return precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks


def compute_classification_metrics(num_matches, num_predictions, num_trgs):
    precision = compute_precision(num_matches, num_predictions)
    recall = compute_recall(num_matches, num_trgs)
    f1 = compute_f1(precision, recall)
    return precision, recall, f1


def compute_precision(num_matches, num_predictions):
    return num_matches / num_predictions if num_predictions > 0 else 0.0


def compute_recall(num_matches, num_trgs):
    return num_matches / num_trgs if num_trgs > 0 else 0.0


def compute_f1(precision, recall):
    return float(2 * (precision * recall)) / (precision + recall) if precision + recall > 0 else 0.0


def update_f1_dict(trg_token_2dlist_stemmed, pred_token_2dlist_stemmed, k_list, f1_dict, tag):
    num_targets = len(trg_token_2dlist_stemmed)
    num_predictions = len(pred_token_2dlist_stemmed)
    is_match = compute_match_result(trg_token_2dlist_stemmed, pred_token_2dlist_stemmed,
                                    type='exact', dimension=1)
    # Classification metrics
    precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks = \
        compute_classification_metrics_at_ks(is_match, num_predictions, num_targets, k_list=k_list,
                                             meng_rui_precision=False)
    for topk, precision_k, recall_k in zip(k_list, precision_ks, recall_ks):
        f1_dict['precision_sum@{}_{}'.format(topk, tag)] += precision_k
        f1_dict['recall_sum@{}_{}'.format(topk, tag)] += recall_k
    return f1_dict


def dcg_at_k(r, k, num_trgs, method=1):
    """
    Reference from https://www.kaggle.com/wendykan/ndcg-example and https://gist.github.com/bwhite/3726239
    Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    num_predictions = r.shape[0]
    if k == 'M':
        k = num_predictions
    elif k == 'G':
        #k = num_trgs
        if num_predictions < num_trgs:
            k = num_trgs
        else:
            k = num_predictions
    elif k == 'O':
        k = num_trgs

    if num_predictions == 0:
        dcg = 0.
    else:
        if num_predictions > k:
            r = r[:k]
            num_predictions = k
        if method == 0:
            dcg = r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            discounted_gain = r / np.log2(np.arange(2, r.size + 2))
            dcg = np.sum(discounted_gain)
        else:
            raise ValueError('method must be 0 or 1.')
    return dcg


def dcg_at_ks(r, k_list, num_trgs, method=1):
    num_predictions = r.shape[0]
    if num_predictions == 0:
        dcg_array = np.array([0] * len(k_list))
    else:
        k_max = -1
        for k in k_list:
            if k == 'M':
                k = num_predictions
            elif k == 'G':
                #k = num_trgs
                if num_predictions < num_trgs:
                    k = num_trgs
                else:
                    k = num_predictions
            elif k == 'O':
                k = num_trgs

            if k > k_max:
                k_max = k
        if num_predictions > k_max:
            r = r[:k_max]
            num_predictions = k_max
        if method == 1:
            discounted_gain = r / np.log2(np.arange(2, r.size + 2))
            dcg = np.cumsum(discounted_gain)
            return_indices = []
            for k in k_list:
                if k == 'M':
                    k = num_predictions
                elif k == 'G':
                    #k = num_trgs
                    if num_predictions < num_trgs:
                        k = num_trgs
                    else:
                        k = num_predictions
                elif k == 'O':
                    k = num_trgs

                return_indices.append((k - 1) if k <= num_predictions else (num_predictions - 1))
            return_indices = np.array(return_indices, dtype=int)
            dcg_array = dcg[return_indices]
        else:
            raise ValueError('method must 1.')
    return dcg_array


def ndcg_at_k(r, k, num_trgs, method=1, include_dcg=False):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    if r.shape[0] == 0:
        ndcg = 0.0
        dcg = 0.0
    else:
        dcg_max = dcg_at_k(np.array(sorted(r, reverse=True)), k, num_trgs, method)
        if dcg_max <= 0.0:
            ndcg = 0.0
        else:
            dcg = dcg_at_k(r, k, num_trgs, method)
            ndcg = dcg / dcg_max
    if include_dcg:
        return ndcg, dcg
    else:
        return ndcg


def ndcg_at_ks(r, k_list, num_trgs, method=1, include_dcg=False):
    if r.shape[0] == 0:
        ndcg_array = [0.0] * len(k_list)
        dcg_array = [0.0] * len(k_list)
    else:
        dcg_array = dcg_at_ks(r, k_list, num_trgs, method)
        ideal_r = np.array(sorted(r, reverse=True))
        dcg_max_array = dcg_at_ks(ideal_r, k_list, num_trgs, method)
        ndcg_array = dcg_array / dcg_max_array
        ndcg_array = np.nan_to_num(ndcg_array)
    if include_dcg:
        return ndcg_array, dcg_array
    else:
        return ndcg_array


def alpha_dcg_at_k(r_2d, k, method=1, alpha=0.5):
    """
    :param r_2d: 2d relevance np array, shape: [num_trg_str, num_pred_str]
    :param k:
    :param method:
    :param alpha:
    :return:
    """
    if r_2d.shape[-1] == 0:
        alpha_dcg = 0.0
    else:
        # convert r_2d to gain vector
        num_trg_str, num_pred_str = r_2d.shape
        if k == 'M':
            k = num_pred_str
        elif k == 'G':
            #k = num_trg_str
            if num_pred_str < num_trg_str:
                k = num_trg_str
            else:
                k = num_pred_str
        elif k == 'O':
            k = num_trg_str
                
        if num_pred_str > k:
            num_pred_str = k
        gain_vector = np.zeros(num_pred_str)
        one_minus_alpha_vec = np.ones(num_trg_str) * (1 - alpha)  # [num_trg_str]
        cum_r = np.concatenate((np.zeros((num_trg_str, 1)), np.cumsum(r_2d, axis=1)), axis=1)
        for j in range(num_pred_str):
            gain_vector[j] = np.dot(r_2d[:, j], np.power(one_minus_alpha_vec, cum_r[:, j]))
        alpha_dcg = dcg_at_k(gain_vector, k, num_trg_str, method)
    return alpha_dcg


def alpha_dcg_at_ks(r_2d, k_list, method=1, alpha=0.5):
    """
    :param r_2d: 2d relevance np array, shape: [num_trg_str, num_pred_str]
    :param ks:
    :param method:
    :param alpha:
    :return:
    """
    if r_2d.shape[-1] == 0:
        return [0.0] * len(k_list)
    # convert r_2d to gain vector
    num_trg_str, num_pred_str = r_2d.shape
    # k_max = max(k_list)
    k_max = -1
    for k in k_list:
        if k == 'M':
            k = num_pred_str
        elif k == 'G':
            #k = num_trg_str
            if num_pred_str < num_trg_str:
                k = num_trg_str
            else:
                k = num_pred_str
        elif k == 'O':
            k = num_trg_str

        if k > k_max:
            k_max = k
    if num_pred_str > k_max:
        num_pred_str = k_max
    gain_vector = np.zeros(num_pred_str)
    one_minus_alpha_vec = np.ones(num_trg_str) * (1 - alpha)  # [num_trg_str]
    cum_r = np.concatenate((np.zeros((num_trg_str, 1)), np.cumsum(r_2d, axis=1)), axis=1)
    for j in range(num_pred_str):
        gain_vector[j] = np.dot(r_2d[:, j], np.power(one_minus_alpha_vec, cum_r[:, j]))
    return dcg_at_ks(gain_vector, k_list, num_trg_str, method)


def alpha_ndcg_at_k(r_2d, k, method=1, alpha=0.5, include_dcg=False):
    """
    :param r_2d: 2d relevance np array, shape: [num_trg_str, num_pred_str]
    :param k:
    :param method:
    :param alpha:
    :return:
    """
    if r_2d.shape[-1] == 0:
        alpha_ndcg = 0.0
        alpha_dcg = 0.0
    else:
        num_trg_str, num_pred_str = r_2d.shape
        if k == 'M':
            k = num_pred_str
        elif k == 'G':
            #k = num_trg_str
            if num_pred_str < num_trg_str:
                k = num_trg_str
            else:
                k = num_pred_str
        elif k == 'O':
            k = num_trg_str
                
        # convert r to gain vector
        alpha_dcg = alpha_dcg_at_k(r_2d, k, method, alpha)
        # compute alpha_dcg_max
        r_2d_ideal = compute_ideal_r_2d(r_2d, k, alpha)
        alpha_dcg_max = alpha_dcg_at_k(r_2d_ideal, k, method, alpha)
        if alpha_dcg_max <= 0.0:
            alpha_ndcg = 0.0
        else:
            alpha_ndcg = alpha_dcg / alpha_dcg_max
            alpha_ndcg = np.nan_to_num(alpha_ndcg)
    if include_dcg:
        return alpha_ndcg, alpha_dcg
    else:
        return alpha_ndcg


def alpha_ndcg_at_ks(r_2d, k_list, method=1, alpha=0.5, include_dcg=False):
    """
    :param r_2d: 2d relevance np array, shape: [num_trg_str, num_pred_str]
    :param k:
    :param method:
    :param alpha:
    :return:
    """
    if r_2d.shape[-1] == 0:
        alpha_ndcg_array = [0] * len(k_list)
        alpha_dcg_array = [0] * len(k_list)
    else:
        # k_max = max(k_list)
        num_trg_str, num_pred_str = r_2d.shape
        k_max = -1
        for k in k_list:
            if k == 'M':
                k = num_pred_str
            elif k == 'G':
                #k = num_trg_str
                if num_pred_str < num_trg_str:
                    k = num_trg_str
                else:
                    k = num_pred_str
            elif k == 'O':
                k = num_trg_str
                    
            if k > k_max:
                k_max = k
        # convert r to gain vector
        alpha_dcg_array = alpha_dcg_at_ks(r_2d, k_list, method, alpha)
        # compute alpha_dcg_max
        r_2d_ideal = compute_ideal_r_2d(r_2d, k_max, alpha)
        alpha_dcg_max_array = alpha_dcg_at_ks(r_2d_ideal, k_list, method, alpha)
        alpha_ndcg_array = alpha_dcg_array / alpha_dcg_max_array
        alpha_ndcg_array = np.nan_to_num(alpha_ndcg_array)
    if include_dcg:
        return alpha_ndcg_array, alpha_dcg_array
    else:
        return alpha_ndcg_array


def compute_ideal_r_2d(r_2d, k, alpha=0.5):
    num_trg_str, num_pred_str = r_2d.shape
    one_minus_alpha_vec = np.ones(num_trg_str) * (1 - alpha)  # [num_trg_str]
    cum_r_vector = np.zeros((num_trg_str))
    ideal_ranking = []
    greedy_depth = min(num_pred_str, k)
    for rank in range(greedy_depth):
        gain_vector = np.zeros(num_pred_str)
        for j in range(num_pred_str):
            if j in ideal_ranking:
                gain_vector[j] = -1000.0
            else:
                gain_vector[j] = np.dot(r_2d[:, j], np.power(one_minus_alpha_vec, cum_r_vector))
        max_idx = np.argmax(gain_vector)
        ideal_ranking.append(max_idx)
        current_relevance_vector = r_2d[:, max_idx]
        cum_r_vector = cum_r_vector + current_relevance_vector
    return r_2d[:, np.array(ideal_ranking, dtype=int)]


def average_precision(r, num_predictions, num_trgs):
    if num_predictions == 0 or num_trgs == 0:
        return 0
    r_cum_sum = np.cumsum(r, axis=0)
    precision_sum = sum([compute_precision(r_cum_sum[k], k + 1) for k in range(num_predictions) if r[k]])
    '''
    precision_sum = 0
    for k in range(num_predictions):
        if r[k] is False:
            continue
        else:
            precision_k = precision(r_cum_sum[k], k+1)
            precision_sum += precision_k
    '''
    return precision_sum / num_trgs


def average_precision_at_k(r, k, num_predictions, num_trgs):
    if k == 'M':
        k = num_predictions
    elif k == 'G':
        #k = num_trgs
        if num_predictions < num_trgs:
            k = num_trgs
        else:
            k = num_predictions
    elif k == 'O':
        k = num_trgs

    if k < num_predictions:
        num_predictions = k
        r = r[:k]
    return average_precision(r, num_predictions, num_trgs)


def average_precision_at_ks(r, k_list, num_predictions, num_trgs):
    if num_predictions == 0 or num_trgs == 0:
        return [0] * len(k_list)
    # k_max = max(k_list)
    k_max = -1
    for k in k_list:
        if k == 'M':
            k = num_predictions
        elif k == 'G':
            #k = num_trgs
            if num_predictions < num_trgs:
                k = num_trgs
            else:
                k = num_predictions
        elif k == 'O':
            k = num_trgs
        if k > k_max:
            k_max = k
    if num_predictions > k_max:
        num_predictions = k_max
        r = r[:num_predictions]
    r_cum_sum = np.cumsum(r, axis=0)
    precision_array = [compute_precision(r_cum_sum[k], k + 1) * r[k] for k in range(num_predictions)]
    precision_cum_sum = np.cumsum(precision_array, axis=0)
    average_precision_array = precision_cum_sum / num_trgs
    return_indices = []
    for k in k_list:
        if k == 'M':
            k = num_predictions
        elif k == 'G':
            #k = num_trgs
            if num_predictions < num_trgs:
                k = num_trgs
            else:
                k = num_predictions
        elif k == 'O':
            k = num_trgs
            
        return_indices.append( (k-1) if k <= num_predictions else (num_predictions-1) )
    return_indices = np.array(return_indices, dtype=int)
    return average_precision_array[return_indices]


def calc_ranking_scores(score_dict, topk_list, present_tag):
    result_list = []
    field_list = []
    for topk in topk_list:
        map_k = sum(score_dict['AP@{}_{}'.format(topk, present_tag)]) / len(
            score_dict['AP@{}_{}'.format(topk, present_tag)])
        # avg_dcg_k = sum(score_dict['DCG@{}_{}'.format(topk, present_tag)]) / len(
        #    score_dict['DCG@{}_{}'.format(topk, present_tag)])
        avg_ndcg_k = sum(score_dict['NDCG@{}_{}'.format(topk, present_tag)]) / len(
            score_dict['NDCG@{}_{}'.format(topk, present_tag)])
        # avg_alpha_dcg_k = sum(score_dict['AlphaDCG@{}_{}'.format(topk, present_tag)]) / len(
        #    score_dict['AlphaDCG@{}_{}'.format(topk, present_tag)])
        avg_alpha_ndcg_k = sum(score_dict['AlphaNDCG@{}_{}'.format(topk, present_tag)]) / len(
            score_dict['AlphaNDCG@{}_{}'.format(topk, present_tag)])
        field_list += ['MAP@{}_{}'.format(topk, present_tag), 'avg_NDCG@{}_{}'.format(topk, present_tag),
                       'AlphaNDCG@{}_{}'.format(topk, present_tag)]
        result_list += [map_k, avg_ndcg_k, avg_alpha_ndcg_k]

    return field_list, result_list


def calc_classification_scores(score_dict, topk_list, present_tag):
    result_list = []
    field_list = []
    for topk in topk_list:
        total_predictions_k = sum(score_dict['num_predictions@{}_{}'.format(topk, present_tag)])
        total_targets_k = sum(score_dict['num_targets@{}_{}'.format(topk, present_tag)])
        total_num_matches_k = sum(score_dict['num_matches@{}_{}'.format(topk, present_tag)])
        # Compute the micro averaged recall, precision and F-1 score
        micro_avg_precision_k, micro_avg_recall_k, micro_avg_f1_score_k = compute_classification_metrics(
            total_num_matches_k, total_predictions_k, total_targets_k)
        # Compute the macro averaged recall, precision and F-1 score
        macro_avg_precision_k = sum(score_dict['precision@{}_{}'.format(topk, present_tag)]) / len(
            score_dict['precision@{}_{}'.format(topk, present_tag)])
        macro_avg_recall_k = sum(score_dict['recall@{}_{}'.format(topk, present_tag)]) / len(
            score_dict['recall@{}_{}'.format(topk, present_tag)])
        macro_avg_f1_score_k = (2 * macro_avg_precision_k * macro_avg_recall_k) / \
                               (macro_avg_precision_k + macro_avg_recall_k) if \
            (macro_avg_precision_k + macro_avg_recall_k) > 0 else 0.0

        field_list += ['micro_avg_p@{}_{}'.format(topk, present_tag), 'micro_avg_r@{}_{}'.format(topk, present_tag),
                       'micro_avg_f1@{}_{}'.format(topk, present_tag),
                       'macro_avg_p@{}_{}'.format(topk, present_tag), 'macro_avg_r@{}_{}'.format(topk, present_tag),
                       'macro_avg_f1@{}_{}'.format(topk, present_tag)]
        result_list += [micro_avg_precision_k, micro_avg_recall_k, micro_avg_f1_score_k,
                        macro_avg_precision_k, macro_avg_recall_k, macro_avg_f1_score_k]
    return field_list, result_list


@gin.configurable
class ExactMatchingMetric(KeyphraseMetric):
    def __init__(self, k_list):
        super(ExactMatchingMetric, self).__init__()
        
        self.k_list = self.__process_input_ks(k_list)
        self.topk_dict = {'present': k_list, 'absent': k_list, 'all': k_list}
        
    def __process_input_ks(self, ks):
        ks_list = []
        for k in ks:
            if k != 'M' and k != 'G' and k != 'O':
                k = int(k)
            ks_list.append(k)
        return ks_list

    def score_corpus(self, all_preds, all_refs, all_inputs):
        source, target, preds = all_inputs, all_refs, all_preds

        score_dict = defaultdict(list)
        self.total_num_src = 0
        self.total_num_src_with_present_keyphrases = 0
        self.total_num_src_with_absent_keyphrases = 0
        self.total_num_unique_predictions = 0
        self.total_num_present_filtered_predictions = 0
        self.total_num_present_unique_targets = 0
        self.total_num_absent_filtered_predictions = 0
        self.total_num_absent_unique_targets = 0
        self.max_unique_targets = 0
        self.sum_incorrect_fraction_for_identifying_present = 0
        self.sum_incorrect_fraction_for_identifying_absent = 0

        predicted_keyphrases = []
        for data_idx, (src_l, trg_l, pred_l) in \
                enumerate(tqdm(zip(source, target, preds),
                            total=len(source), desc='Evaluating...')):
            self.total_num_src += 1
            
            src_token_list, stemmed_src_token_list = src_l 
            pred_str_list, pred_token_2dlist, stemmed_pred_token_2dlist = pred_l
            trg_str_list, trg_token_2dlist, stemmed_trg_token_2dlist = trg_l
            num_predictions = len(pred_str_list)

            # Filter out duplicate or invalid predictions
            filtered_stemmed_pred_token_2dlist, num_duplicated_predictions, is_unique_mask = filter_prediction(
                self.invalidate_unk, stemmed_pred_token_2dlist, self.unk_word
            )
            self.total_num_unique_predictions += (num_predictions - num_duplicated_predictions)
            num_filtered_predictions = len(filtered_stemmed_pred_token_2dlist)

            # Remove duplicated targets
            unique_stemmed_trg_token_2dlist, num_duplicated_trg = find_unique_target(stemmed_trg_token_2dlist)
            num_unique_targets = len(unique_stemmed_trg_token_2dlist)

            if num_unique_targets > self.max_unique_targets:
                self.max_unique_targets = num_unique_targets

            # separate present and absent keyphrases
            present_filtered_stemmed_pred_token_2dlist, absent_filtered_stemmed_pred_token_2dlist, is_present_mask = \
                separate_present_absent_by_source(stemmed_src_token_list, filtered_stemmed_pred_token_2dlist, False)
            present_unique_stemmed_trg_token_2dlist, absent_unique_stemmed_trg_token_2dlist, _ = \
                separate_present_absent_by_source(stemmed_src_token_list, unique_stemmed_trg_token_2dlist, False)

            # save the predicted keyphrases
            filtered_pred_token_2dlist = [kp_tokens for kp_tokens, is_unique
                                        in zip(pred_token_2dlist, is_unique_mask) if is_unique]
            result = {'id': data_idx, 'present': [], 'absent': []}
            for kp_tokens, is_present in zip(filtered_pred_token_2dlist, is_present_mask):
                if is_present:
                    result['present'] += [' '.join(kp_tokens)]
                else:
                    result['absent'] += [' '.join(kp_tokens)]
            predicted_keyphrases.append(result)
            
            self.total_num_present_filtered_predictions += len(present_filtered_stemmed_pred_token_2dlist)
            self.total_num_present_unique_targets += len(present_unique_stemmed_trg_token_2dlist)
            self.total_num_absent_filtered_predictions += len(absent_filtered_stemmed_pred_token_2dlist)
            self.total_num_absent_unique_targets += len(absent_unique_stemmed_trg_token_2dlist)
            if len(present_unique_stemmed_trg_token_2dlist) > 0:
                self.total_num_src_with_present_keyphrases += 1
            if len(absent_unique_stemmed_trg_token_2dlist) > 0:
                self.total_num_src_with_absent_keyphrases += 1

            # compute all the metrics and update the score_dict
            score_dict = update_score_dict(unique_stemmed_trg_token_2dlist, filtered_stemmed_pred_token_2dlist,
                                           self.topk_dict['all'], score_dict, 'all')
            # compute all the metrics and update the score_dict for present keyphrase
            score_dict = update_score_dict(present_unique_stemmed_trg_token_2dlist,
                                           present_filtered_stemmed_pred_token_2dlist,
                                           self.topk_dict['present'], score_dict, 'present')
            # compute all the metrics and update the score_dict for absent keyphrase
            score_dict = update_score_dict(absent_unique_stemmed_trg_token_2dlist,
                                           absent_filtered_stemmed_pred_token_2dlist,
                                           self.topk_dict['absent'], score_dict, 'absent')

        return score_dict

    def aggregate_scores(self, score_dict):      
        # classification
        field_list_all, result_list_all = calc_classification_scores(score_dict, self.k_list, 'all')
        field_list_present, result_list_present = calc_classification_scores(score_dict, self.k_list, 'present')
        field_list_absent, result_list_absent = calc_classification_scores(score_dict, self.k_list, 'absent')
        field_list_classification = field_list_all + field_list_present + field_list_absent
        result_list_classification = result_list_all + result_list_present + result_list_absent
        
        # ranking metrics
        field_list_all, result_list_all = calc_ranking_scores(score_dict, self.k_list, 'all')
        field_list_present, result_list_present = calc_ranking_scores(score_dict, self.k_list, 'present')
        field_list_absent, result_list_absent = calc_ranking_scores(score_dict, self.k_list, 'absent')
        field_list_ranking = field_list_all + field_list_present + field_list_absent
        result_list_ranking = result_list_all + result_list_present + result_list_absent
        
        aggregated_scores_dict = {k: v for k, v in zip(field_list_classification + field_list_ranking, 
                                                       result_list_classification + result_list_ranking)}
        
        return score_dict, aggregated_scores_dict
