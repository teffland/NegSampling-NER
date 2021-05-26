from typing import *

import numpy as np

import torch
from torch import nn
from transformers import AutoModel
from collections import defaultdict

from misc import flat_list
from misc import iterative_support, conflict_judge
from utils import UnitAlphabet, LabelAlphabet, BatchTooLargeException, BatchTooLongException



class PhraseClassifier(nn.Module):
    def __init__(
        self,
        lexical_vocab: UnitAlphabet,
        label_vocab: LabelAlphabet,
        hidden_dim: int,
        dropout_rate: float,
        neg_rate: float,
        bert_path: str,
    ):
        super(PhraseClassifier, self).__init__()

        self._lexical_vocab = lexical_vocab
        self._label_vocab = label_vocab
        self._neg_rate = neg_rate

        self._encoder = BERT(bert_path)
        self._classifier = MLP(self._encoder.dimension * 4, hidden_dim, len(label_vocab), dropout_rate)
        self._criterion = nn.NLLLoss()

    def forward(self, var_h, **kwargs):
        
        con_repr = self._encoder(var_h, kwargs["mask_mat"], kwargs["starts"])

        batch_size, token_num, hidden_dim = con_repr.size()
        # print(f'B, L, H: {batch_size}, {token_num}, {hidden_dim}')
        ext_row = con_repr.unsqueeze(2).expand(batch_size, token_num, token_num, hidden_dim)
        ext_column = con_repr.unsqueeze(1).expand_as(ext_row)
        table = torch.cat([ext_row, ext_column, ext_row - ext_column, ext_row * ext_column], dim=-1)
        return self._classifier(table)

    def _align_bpes(self, tokens: List[str], offset_mapping: List[Tuple[int, int]]):
        # From transformers_converter
        """ Return a multi-mapping from token indices to subword indices"""
        # print("ALIGN")
        # print(f"Tokens: {tokens}")
        # print(f"Offset mapping: {offset_mapping}")
        # Mapping from char idxs in joined text str to original tokens
        charidx2tokidx = {}
        for i, t in enumerate(tokens):
            offset = len(charidx2tokidx) + i
            for k in range(len(t)):
                charidx2tokidx[offset + k] = i

        # print(f"char2tok", charidx2tokidx)

        # Mapping derived from the offset mapping to go brom token idx to bpe idx
        tokidx2bpeidx = defaultdict(list)
        for i, (s, e) in enumerate(offset_mapping):
            # print(i, (s, e))
            if s == e:
                continue  # skip special tokens
            # if s > max(charidx2tokidx.keys()):
            #     break
            else:
                tokidx = charidx2tokidx[s]
                tokidx2bpeidx[tokidx].append(i)
        return tokidx2bpeidx

    def _pre_process_input(self, utterances):
        """ Edited to use AutoModel/AutoTokenizer and split long sentences. 
        The original paper only worked on datasets where sentences were always short enough/in english,
        so they didn't need this.
        """

        # lengths = [len(s) for s in utterances]
        # max_len = max(lengths)
        # pieces = iterative_support(self._lexical_vocab.tokenize, utterances)
        # print("pieces", pieces)
        # units, positions = [], []

        # for tokens in pieces:
        #     units.append(flat_list(tokens))
        #     cum_list = np.cumsum([len(p) for p in tokens]).tolist()
        #     positions.append([0] + cum_list[:-1])

        # sizes = [len(u) for u in units]
        # max_size = max(sizes)
        # cls_sign = self._lexical_vocab.CLS_SIGN
        # sep_sign = self._lexical_vocab.SEP_SIGN
        # pad_sign = self._lexical_vocab.PAD_SIGN
        # pad_unit = [[cls_sign] + s + [sep_sign] + [pad_sign] * (max_size - len(s)) for s in units]
        # starts = [[ln + 1 for ln in u] + [max_size + 1] * (max_len - len(u)) for u in positions]

        # var_unit = torch.LongTensor([self._lexical_vocab.index(u) for u in pad_unit])
        # attn_mask = torch.LongTensor([[1] * (lg + 2) + [0] * (max_size - lg) for lg in sizes])
        # var_start = torch.LongTensor(starts)

        # Use the updated version of transformers tokenizer that does all of the above for us
        # print("\nTokenizing: ", utterances)
        lengths = [len(s) for s in utterances]
        utterances = [
            " ".join(tokens) for tokens in utterances
        ]  # we have to join them otherwise RoBERTa will break. An idiosyncrasy of 3.5.1
        out = self._lexical_vocab._tokenizer(
            utterances,
            is_split_into_words=False,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        # print("\noutput:", out.keys())
        var_unit = out["input_ids"]
        batch_size, max_size = var_unit.shape
        print(f'\nBatch size, seq size: {var_unit.shape}', flush=True)
        if max_size > 190 and batch_size > 1:
            # The gpu can't handle very long sequences of size for batches larger than one
            # The N^2 causes Cuda OOM issues, so in these cases we back off to smaller batch sizes.
            raise BatchTooLargeException()

        max_len = max(lengths)
        attn_mask = out["attention_mask"]
        offset_mappings = out["offset_mapping"]
        var_start = []
        for utt_i, (utterance, offset_mapping) in enumerate(zip(utterances, offset_mappings)):
            tokens = utterance.split()
            tokidx2bpeidx = self._align_bpes(tokens, offset_mapping.numpy().tolist())
            L = len(tokens)
            if len(tokidx2bpeidx) < L:
                raise BatchTooLongException()
                # print(f'WARNING: Sentence is too long with -- chopping it off at max size')
                # print('len toks', len(tokens))
                # print("tok2bpe", tokidx2bpeidx)
                # L = len(tokidx2bpeidx)
                # lengths[utt_i] = L
            var_start.append(
                [tokidx2bpeidx[i][0] for i in range(L)] + [max_size - 1] * (max_size - L)
            )
        var_start = torch.LongTensor(var_start)

        # print(f"var_unit: {var_unit.shape}, attn_mask: {attn_mask.shape}, var_start: {var_start.shape}") 

        if torch.cuda.is_available():
            var_unit = var_unit.cuda()
            attn_mask = attn_mask.cuda()
            var_start = var_start.cuda()

        return var_unit, attn_mask, var_start, lengths

    def _pre_process_output(self, entities, lengths):
        positions, labels = [], []
        batch_size = len(entities)

        for utt_i in range(0, batch_size):
            for segment in entities[utt_i]:
                positions.append((utt_i, segment[0], segment[1]))
                labels.append(segment[2])

        for utt_i in range(0, batch_size):
            reject_set = [(e[0], e[1]) for e in entities[utt_i]]
            s_len = lengths[utt_i]
            neg_num = int(s_len * self._neg_rate) + 1

            candies = flat_list([[(i, j) for j in range(i, s_len) if (i, j) not in reject_set] for i in range(s_len)])
            if len(candies) > 0:
                sample_num = min(neg_num, len(candies))
                assert sample_num > 0

                np.random.shuffle(candies)
                for i, j in candies[:sample_num]:
                    positions.append((utt_i, i, j))
                    labels.append("O")

        var_lbl = torch.LongTensor(iterative_support(self._label_vocab.index, labels))
        if torch.cuda.is_available():
            var_lbl = var_lbl.cuda()
        return positions, var_lbl

    def estimate(self, sentences, segments):
        var_sent, attn_mask, start_mat, lengths = self._pre_process_input(sentences)
        score_t = self(var_sent, mask_mat=attn_mask, starts=start_mat)

        positions, targets = self._pre_process_output(segments, lengths)
        flat_s = torch.cat([score_t[[i], j, k] for i, j, k in positions], dim=0)
        return self._criterion(torch.log_softmax(flat_s, dim=-1), targets)

    def inference(self, sentences):
        # print("inference!")
        var_sent, attn_mask, starts, lengths = self._pre_process_input(sentences)
        log_items = self(var_sent, mask_mat=attn_mask, starts=starts)

        score_t = torch.log_softmax(log_items, dim=-1)
        val_table, idx_table = torch.max(score_t, dim=-1)

        listing_it = idx_table.cpu().numpy().tolist()
        listing_vt = val_table.cpu().numpy().tolist()
        label_table = iterative_support(self._label_vocab.get, listing_it)

        candidates = []
        for l_mat, v_mat, sent_l in zip(label_table, listing_vt, lengths):
            candidates.append([])
            for i in range(0, sent_l):
                for j in range(i, sent_l):
                    if l_mat[i][j] != "O":
                        candidates[-1].append((i, j, l_mat[i][j], v_mat[i][j]))
                        # print(f"\n\nadd ent {candidates[-1]} for sent len {sent_l}")

        entities = []
        for segments in candidates:
            ordered_seg = sorted(segments, key=lambda e: -e[-1])
            filter_list = []
            for elem in ordered_seg:
                flag = False
                current = (elem[0], elem[1])
                for prior in filter_list:
                    flag = conflict_judge(current, (prior[0], prior[1]))
                    if flag:
                        break
                if not flag:
                    filter_list.append((elem[0], elem[1], elem[2]))
            entities.append(sorted(filter_list, key=lambda e: e[0]))
        # print("entities", entities)
        return entities


class BERT(nn.Module):
    def __init__(self, source_path):
        super(BERT, self).__init__()
        self.bert = AutoModel.from_pretrained(source_path)

    @property
    def dimension(self):
        return 768

    def forward(self, input_ids, attn_mask, starts):
        all_hidden = self.bert(input_ids=input_ids, attention_mask=attn_mask)[0]

        batch_size, _, hidden_dim = all_hidden.size()
        _, unit_num = starts.size()
        positions = starts.unsqueeze(-1).expand(batch_size, unit_num, hidden_dim)
        return torch.gather(all_hidden, dim=-2, index=positions)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(MLP, self).__init__()

        self._activator = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, output_dim))
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, var_h):
        return self._activator(self._dropout(var_h))
