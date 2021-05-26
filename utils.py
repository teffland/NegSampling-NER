from typing import *

from tqdm import tqdm
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from misc import extract_json_data
from misc import iob_tagging, f1_score

class BatchTooLargeException(Exception):
    pass

class BatchTooLongException(Exception):
    pass

class UnitAlphabet(object):

    CLS_SIGN, SEP_SIGN = "[CLS]", "[SEP]"
    PAD_SIGN, UNK_SIGN = "[PAD]", "[UNK]"

    def __init__(self, model_name):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize(self, item):
        return self._tokenizer.tokenize(item)

    def index(self, items):
        return self._tokenizer.convert_tokens_to_ids(items)


class LabelAlphabet(object):
    def __init__(self):
        super(LabelAlphabet, self).__init__()

        self._idx_to_item = []
        self._item_to_idx = {}

    def add(self, item):
        if item not in self._item_to_idx:
            self._item_to_idx[item] = len(self._idx_to_item)
            self._idx_to_item.append(item)

    def get(self, idx):
        return self._idx_to_item[idx]

    def index(self, item):
        return self._item_to_idx[item]

    def __str__(self):
        return str(self._item_to_idx)

    def __len__(self):
        return len(self._idx_to_item)


def corpus_to_iterator(file_path, batch_size, if_shuffle, label_vocab=None):
    material = extract_json_data(file_path)
    instances = [(e["sentence"], e["labeled entities"]) for e in material]

    if label_vocab is not None:
        label_vocab.add("O")
        for _, u in instances:
            for _, _, l in u:
                label_vocab.add(l)

    class _DataSet(Dataset):
        def __init__(self, elements):
            self._elements = elements

        def __getitem__(self, item):
            return self._elements[item]

        def __len__(self):
            return len(self._elements)

    def distribute(elements):
        sentences, entities = [], []
        for s, e in elements:
            sentences.append(s)
            entities.append(e)
        return sentences, entities

    wrap_data = _DataSet(instances)
    return DataLoader(wrap_data, batch_size, if_shuffle, collate_fn=distribute)


class Procedure(object):
    @staticmethod
    def train(model, dataset, optimizer):
        model.train()
        time_start, total_penalties = time.time(), 0.0

        def step(batch, total_penalties):
            loss = model.estimate(*batch)
            total_penalties += loss.cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for batch in tqdm(dataset, ncols=50):
            try:
                step(batch, total_penalties)
            except BatchTooLargeException as e:
                print('Batch too large, breaking up into single sentence batches...', flush=True)
                for sentence, segment in zip(*batch):
                    try:
                        step(([sentence], [segment]), total_penalties)
                    except BatchTooLongException:
                        print('Batch too long, breaking up long sentence')
                        for sent, seg in zip(*breakup_sentence(sentence, segment)):
                            step(([sent], [seg]), total_penalties)
                print('Done single-sentence batching', flush=True)

        time_con = time.time() - time_start
        return total_penalties, time_con

    @staticmethod
    def test(model, dataset, eval_path):
        model.eval()
        time_start = time.time()
        seqs, outputs, oracles = [], [], []

        for sentences, segments in tqdm(dataset, ncols=50):
            def step(sentences, segments):
                with torch.no_grad():
                    predictions = model.inference(sentences)

                seqs.extend(sentences)
                outputs.extend([iob_tagging(e, len(u)) for e, u in zip(predictions, sentences)])
                oracles.extend([iob_tagging(e, len(u)) for e, u in zip(segments, sentences)])

            try:
                step(sentences, segments)
            except BatchTooLargeException as e:
                print('Batch too large, breaking up into single sentence batches...', flush=True)
                for sentence, segment in zip(sentences, segments):
                    try:
                        step([sentence], [segment])
                    except BatchTooLongException:
                        print('Batch too long, breaking up long sentence')
                        for sent, seg in zip(*breakup_sentence(sentence, segment)):
                            step([sent], [seg])

                print('Done single-sentence batching', flush=True)


        out_f1 = f1_score(seqs, outputs, oracles, eval_path)
        return out_f1, time.time() - time_start

def breakup_sentence(sentence, segment, max_tokens=50):
    """ Break a too-long sentence up into processable chunks. """
    segment = sorted(segment, key=lambda x:x[1]) # sorted by end position
    sentences, segments = [], []
    s = 0
    while s < len(sentence):
        chunk_segments = []
        e = min(s + max_tokens, len(sentence)-1)
        for (l, r, t) in segment:
            if s <= l and l <= e:
                # ent starts in chunk
                if r <= e:
                    # ent fits in chunk
                    chunk_segments.append((l-s, r-s, t))
                else:
                    # ent crosses chunk boundary, so omit the ent and shorten the chunk to be in front of the entity
                    e -= e-l+1
                    break
            elif l > e:
                # the rest of the entities are to the right of this chunk
                break 
        
        sentences.append(sentence[s:e+1])
        segments.append(chunk_segments)
        s = e+1

    print(f' Sentence before: {sentence}')
    print(f' Entities before: {[ (sentence[l:r+1], t) for l,r,t in segment]}')
    print('After:')
    for i in range(len(sentences)):
         print(f' sentence {i} after: {sentences[i]}')
         print(f' entities {i} after: {[ (sentences[i][l:r+1], t) for l,r,t in segments[i]]}')
    return sentences, segments
        