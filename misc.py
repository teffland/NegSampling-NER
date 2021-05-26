import time
import json
import codecs
import os
import numpy as np
import random

import torch


def iterative_support(func, query):
    if isinstance(query, (list, tuple, set)):
        return [iterative_support(func, i) for i in query]
    return func(query)


def fix_random_seed(state_val):
    random.seed(state_val)
    np.random.seed(state_val)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(state_val)
        torch.cuda.manual_seed_all(state_val)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.manual_seed(state_val)
    torch.random.manual_seed(state_val)


def flat_list(h_list):
    e_list = []

    for item in h_list:
        if isinstance(item, list):
            e_list.extend(flat_list(item))
        else:
            e_list.append(item)
    return e_list


from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
from collections import defaultdict


def f1_score(sent_list, pred_list, gold_list, script_path):
    """Change their f1_score to use allennlp f1 so we know they are comparable.

    They already provide a list of IOB lists, one per sentence, so all we need to do is convert to spans
    and check for precision, recall, f1
    See: https://github.com/allenai/allennlp/blob/v1.2.1/allennlp/training/metrics/span_based_f1_measure.py#L190-L204
    """
    tags_to_spans_function = bio_tags_to_spans

    def compute_metrics(true_positives, false_positives, false_negatives):
        precision = true_positives / (true_positives + false_positives + 1e-13)
        recall = true_positives / (true_positives + false_negatives + 1e-13)
        f1_measure = 2.0 * (precision * recall) / (precision + recall + 1e-13)
        return precision, recall, f1_measure

    # Gather stats
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)

    for predicted_string_labels, gold_string_labels in zip(pred_list, gold_list):
        # print("gold tags", gold_string_labels)
        # print("pred tags", predicted_string_labels)
        predicted_spans = tags_to_spans_function(predicted_string_labels)
        gold_spans = tags_to_spans_function(gold_string_labels)

        for span in predicted_spans:
            if span in gold_spans:
                true_positives[span[0]] += 1
                gold_spans.remove(span)
            else:
                false_positives[span[0]] += 1
        # These spans weren't predicted.
        for span in gold_spans:
            false_negatives[span[0]] += 1

    # Calc metrics
    all_tags = set()
    all_tags.update(true_positives.keys())
    all_tags.update(false_positives.keys())
    all_tags.update(false_negatives.keys())
    all_metrics = {}
    for tag in all_tags:
        precision, recall, f1_measure = compute_metrics(
            true_positives[tag], false_positives[tag], false_negatives[tag]
        )
        precision_key = "precision" + "-" + tag
        recall_key = "recall" + "-" + tag
        f1_key = "f1-measure" + "-" + tag
        all_metrics[precision_key] = precision
        all_metrics[recall_key] = recall
        all_metrics[f1_key] = f1_measure

    # Compute the precision, recall and f1 for all spans jointly.
    precision, recall, f1_measure = compute_metrics(
        sum(true_positives.values()),
        sum(false_positives.values()),
        sum(false_negatives.values()),
    )
    all_metrics["micro-precision"] = precision
    all_metrics["micro-recall"] = recall
    all_metrics["micro-f1"] = f1_measure
    print(f"ALL METRICS:\n{json.dumps(all_metrics, indent=2, sort_keys=True)}")

    return f1_measure

    # fn_out = 'eval_%04d.txt' % random.randint(0, 10000)
    # if os.path.isfile(fn_out):
    #     os.remove(fn_out)

    # text_file = open(fn_out, mode='w')
    # for i, words in enumerate(sent_list):
    #     tags_1 = gold_list[i]
    #     tags_2 = pred_list[i]
    #     for j, word in enumerate(words):
    #         tag_1 = tags_1[j]
    #         tag_2 = tags_2[j]
    #         text_file.write('%s %s %s\n' % (word, tag_1, tag_2))
    #     text_file.write('\n')
    # text_file.close()

    # cmd = 'perl %s < %s' % (script_path, fn_out)
    # msg = '\nStandard CoNNL perl script (author: Erik Tjong Kim Sang <erikt@uia.ua.ac.be>, version: 2004-01-26):\n'
    # msg += ''.join(os.popen(cmd).readlines())
    # time.sleep(1.0)
    # if fn_out.startswith('eval_') and os.path.exists(fn_out):
    #     os.remove(fn_out)
    # return float(msg.split('\n')[3].split(':')[-1].strip())


def iob_tagging(entities, s_len):
    tags = ["O"] * s_len
    # print("slen", s_len)

    for el, er, et in entities:
        # print("entity:", el, er, et)
        for i in range(el, er + 1):
            if i == el:
                tags[i] = "B-" + et
            else:
                tags[i] = "I-" + et
    return tags


def conflict_judge(line_x, line_y):
    if line_x[0] == line_y[0]:
        return True
    if line_x[0] < line_y[0]:
        if line_x[1] >= line_y[0]:
            return True
    if line_x[0] > line_y[0]:
        if line_x[0] <= line_y[1]:
            return True
    return False


def extract_json_data(file_path):
    with codecs.open(file_path, "r", "utf-8") as fr:
        dataset = json.load(fr)  # [:500]  # short dataset for testing
    return dataset
