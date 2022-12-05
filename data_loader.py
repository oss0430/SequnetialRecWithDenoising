import numpy as np
from numpy.random import permutation, poisson
import torch
import random
import copy
import math
import pickle

from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

from dataclasses import dataclass
from typing import Dict, List, Optional
import pdb
from config import *
from sklearn.model_selection import train_test_split

class AmazonData:
    def __init__(
        self,
        config,
    ):
        self.config = config
        self.data = data = DFDATASET
        self.review_user   = data['user_id']
        self.review_item   = data['item_id']

        """
        Used data format:
        {
            'user_id': {
                'input_sequence'    : list(int),    ## [1] - [3] - [4] - [8]
                'positive_sequence' : list(int),    ## [1] - [3] - [4] - [8] -[11]
                'negative_sequence' : list(int),    ## [77] - [123] - [2534] - [22] - [52]
                'target_item'       : int           ## model will predict [11] from input_sequence
            }
        }
        """
        
        user_set = set(data['user_id'].values.tolist())
        item_set = set(data['item_id'].values.tolist())

        user_length, item_length = len(user_set), len(item_set)

        # Split data with user set 80:10:10
        train_size = 0.8
        valid_size = 0.1

        train_index = int(user_length * train_size)
        train_user = list(user_set)[0:train_index]
        rem_user = list(user_set)[train_index:]

        valid_index = int(user_length * valid_size)
        valid_user = list(user_set)[train_index:train_index+valid_index]
        test_user = list(user_set)[train_index+valid_index:]
        
        try:
            with open('data/train.pkl', 'rb') as handle:
                self.train = train = pickle.load(handle)
            with open('data/valid.pkl', 'rb') as handle:
                self.valid = valid = pickle.load(handle)
            with open('data/test.pkl', 'rb') as handle:
                self.test = test = pickle.load(handle)
                
        except:
            print("Start data preprocessing....")
            self.train = train = []
            self.valid = valid = []
            self.test = test = []

            usernum, itemnum = 0, 0
            user_item_list = defaultdict(list)
            for idx, row in data.iterrows():
                u = row['user_id']
                i = row['item_id']

                usernum = max(u, usernum)
                itemnum = max(i, itemnum)
                user_item_list[u].append(i)

            for user, item_sequence in user_item_list.items():

                # Mask last item of input sequence
                input_sequence = item_sequence[:-1]
                # masked_sequence = input_sequence.append(-100)

                if user in train_user:
                    train.append({'user_id': user, 'input_sequence': input_sequence, 'positive_sequence': item_sequence, \
                        'negative_sequence': random.sample(item_set, len(item_sequence)), 'target_item': item_sequence[-1]})
                elif user in valid_user:
                    valid.append({'user_id': user, 'input_sequence': input_sequence, 'positive_sequence': item_sequence, \
                        'negative_sequence': random.sample(item_set, len(item_sequence)), 'target_item': item_sequence[-1]})
                else:
                    test.append({'user_id': user, 'input_sequence': input_sequence, 'positive_sequence': item_sequence, \
                        'negative_sequence': random.sample(item_set, len(item_sequence)), 'target_item': item_sequence[-1]})

            with open('data/train.pkl', 'wb') as handle:
                pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('data/valid.pkl', 'wb') as handle:
                pickle.dump(valid, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('data/test.pkl', 'wb') as handle:
                pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.config.usernum = self.usernum = len(user_set)
        self.config.itemnum = self.itemnum = len(item_set)
        self.config.n_train = len(train)
        self.config.n_valid = len(valid)
        self.config.n_test = len(test)
    
    def _random_neq(
        self,
        l,
        r,
        s
    ):
        t = random.randint(l, r)
        while t in s:
            ## if random negative item is in set S
            ## Generate a new one
            t = random.randint(l, r)
        return t

    def get_data(self, mode):
        if mode == "train":
            return self.train, len(self.train)
        elif mode == "valid":
            return self.valid, len(self.valid)
        elif mode == "test":
            return self.test, len(self.test)
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()


class SeqRecDataset(Dataset):
    def __init__(
        self,
        config,
        mode,
        segment_seq = False,
        segment_length = 5
    ):
        self.config = config
        self.mode = mode
        self.segment_seq = segment_seq
        self.segment_length = segment_length
        self.max_len = config.max_lengths

        dataset = AmazonData(config)
        self.data, self.len = dataset.get_data(mode)

        self.usernum = usernum = config.usernum
        self.itemnum = itemnum = config.itemnum

        self.padding_idx = 0
        self.bos_idx = itemnum + 2
        self.eos_idx = itemnum + 3
    
    def __len__(self):
        return self.len
    
    def _get_101_candidate_items(
        self,
        target_item
    ):
        candidate_item = []

        candidate_item.append(target_item)
        for _ in range(100):
            item_id = random.randint(1,self.itemnum)
            while target_item == item_id :
                item_id = random.randint(1,self.itemnum)
            candidate_item.append(item_id)

        return candidate_item

    def _pad_and_trunc_by_max_len(
        self,
        sequence = []
    ):
        ## sequence = [1] - [2] - [3]
        ## max_len  = 5
        ## return   = [0] - [1] - [2] - [3] - [eos]
        
        ## if permutatin
        ## seuence = [1] - [2] - [3] - [4] - [5]
        ## segment_len = 2
        ## return = [1]  - [2] - [eos]  - [3] - [4] - [eos] - [5] - [eos]
        
        
        if self.segment_seq :
            segment_length = self.segment_length
            idx = [i for i in range(segment_length - 1, len(sequence) - 1, segment_length)]
            sequence = np.insert(sequence, idx, self.eos_idx, axis = 0).tolist()

        #sequence = sequence + [self.eos_idx]
        ## NEW NOW item starts at first
        if len(sequence) > self.max_len :
            sequence = sequence[len(sequence):]
        
        new_sequence = sequence + [self.padding_idx] * self.max_len 

        return new_sequence[:self.max_len]

    def _pad_and_trunc_by_max_len_for_pos(
        self,
        sequence = []
    ):
        ## sequence = [1] - [2] - [3]
        ## max_len  = 5
        ## return   = [0] - [1] - [2] - [3] - [eos]
        
        ## if permutatin
        ## seuence = [1] - [2] - [3] - [4] - [5]
        ## segment_len = 2
        ## return = [1]  - [2] - [eos]  - [3] - [4] - [eos] - [5] - [eos]
        
        
        if self.segment_seq :
            segment_length = self.segment_length
            idx = [i for i in range(segment_length - 1, len(sequence) - 1, segment_length)]
            sequence = np.insert(sequence, idx, self.eos_idx, axis = 0).tolist()

        #sequence = sequence + [self.eos_idx]
        ## NEW NOW item starts at first
        if len(sequence) > self.max_len :
            sequence = sequence[len(sequence):]
        
        new_sequence = sequence + [-100] * self.max_len 

        return new_sequence[:self.max_len]
    
    def __getitem__(self, index=None):
        for idx in range(len(self.data)):
            user_id = np.array(self.data[idx]['user_id'])
            input_ids = np.array(self._pad_and_trunc_by_max_len(self.data[idx]['input_sequence']))
            positive_ids = np.array(self._pad_and_trunc_by_max_len(self.data[idx]['positive_sequence']))
            negative_ids = np.array(self._pad_and_trunc_by_max_len(self.data[idx]['negative_sequence']))
            target_item = np.array(self._pad_and_trunc_by_max_len([self.data[idx]['target_item']]))

        return {
            'user_id'        : user_id,
            'input_ids'      : input_ids,
            'positive_ids'   : positive_ids,
            'negative_ids'   : negative_ids,
            'target_item'    : target_item
        }
    
def get_loader(config, mode, shuffle=True):
    dataset = SeqRecDataset(config, mode)

    print(mode)

    if mode == 'train':
        config.n_train = len(dataset)
        batch_size = config.train_batch_size
    elif mode == 'valid':
        config.n_valid = len(dataset)
        batch_size = config.valid_batch_size
    elif mode == 'test':
        config.n_test = len(dataset)
        batch_size = config.valid_batch_size

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


@dataclass
class DataCollatorForDenoisingTasks(object):
    """Data collator used denoising language modeling task in BART.
    The implementation is based on
    https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/fairseq/data/denoising_dataset.py.
    The default paramters is based on BART paper https://arxiv.org/abs/1910.13461.
    """

    def __init__(
        self,
        mask_ratio: float,
        poisson_lambda: float,
        permutate_sentence_ratio: float,
        eos_token_id: int,
        bos_token_id: int,
        pad_token_id: int,
        mask_token_id: int,
        pad_to_multiple_of: int
    ):
        self.mask_ratio = mask_ratio
        self.poisson_lambda = poisson_lambda
        self.permutate_sentence_ratio = permutate_sentence_ratio
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        

    def __call__(self, examples: List[Dict[str, List[torch.Tensor]]]) -> Dict[str, np.ndarray]:
        """Batching, adding whole word mask and permutate sentences
        Args:
            examples (dict): list of examples each examples contains input_ids field
        """
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = dict()
        batch['user_ids'] = np.stack([i['input_ids'] for i in examples], axis = 0)
        batch['input_ids'] = np.stack([i['input_ids'] for i in examples], axis = 0)
        batch["decoder_input_ids"] = self.shift_tokens_right(batch["input_ids"])
        do_permutate = False
        if self.permutate_sentence_ratio > 0.0:
            batch["input_ids"] = self.permutate_sentences(batch["input_ids"])
            do_permutate = True

        if self.mask_ratio:
            batch["input_ids"], batch["labels"] = self.add_whole_word_mask(batch["input_ids"], do_permutate)
        return batch

    def shift_tokens_right(self, inputs):
        """Shift decoder input ids right: https://github.com/huggingface/transformers/issues/7961.
        Examples:
            <s>My dog is cute.</s><s>It loves to play in the park.</s><pad><pad>
            shift to -> </s><s>My dog is cute.</s><s>It loves to play in the park.<pad><pad>
        """

        shifted_inputs = np.roll(inputs, 1, axis=-1)

        # replace first token with eos token
        shifted_inputs[:, 0] = self.eos_token_id

        # when there's padding, the last eos tokens will not be rotate to first positon
        # we'll need to replace it with a padding token

        # replace eos tokens at the end of sequences with pad tokens
        end_with_eos = np.where(shifted_inputs[:, -1] == self.eos_token_id)
        shifted_inputs[end_with_eos, -1] = self.pad_token_id

        # find positions where where's the token is eos and its follwing token is a padding token
        last_eos_indices = np.where(
            (shifted_inputs[:, :-1] == self.eos_token_id)
            * (shifted_inputs[:, 1:] == self.pad_token_id)
        )

        # replace eos tokens with pad token
        shifted_inputs[last_eos_indices] = self.pad_token_id
        return shifted_inputs

    def permutate_sentences(self, inputs):
        results = inputs.copy()
        full_stops = inputs == self.eos_token_id
        sentence_ends = np.argwhere(full_stops[:, 1:] * ~full_stops[:, :-1])
        sentence_ends[:, 1] += 2
        num_sentences = np.unique(sentence_ends[:, 0], return_counts=True)[1]
        num_to_permute = np.ceil((num_sentences * 2 * self.permutate_sentence_ratio) / 2.0).astype(int)

        sentence_ends = np.split(sentence_ends[:, 1], np.unique(sentence_ends[:, 0], return_index=True)[1][1:])

        for i in range(inputs.shape[0]):
            substitutions = np.random.permutation(num_sentences[i])[: num_to_permute[i]]

            ordering = np.arange(0, num_sentences[i])
            ordering[substitutions] = substitutions[np.random.permutation(num_to_permute[i])]

            index = 0
            for j in ordering:
                sentence = inputs[i, (sentence_ends[i][j - 1] if j > 0 else 0) : sentence_ends[i][j]]
                results[i, index : index + sentence.shape[0]] = sentence
                index += sentence.shape[0]
        
        return results

    def add_whole_word_mask(self, inputs, do_permutate):
        labels = inputs.copy()
        special_tokens_mask = (labels == self.eos_token_id) & (labels == self.bos_token_id)

        # determine how many tokens we need to mask in total
        is_token = ~(labels == self.pad_token_id) & ~special_tokens_mask
        num_to_mask = int(math.ceil(is_token.astype(float).sum() * self.mask_ratio))
        if num_to_mask == 0:
            return inputs, labels

        # generate a sufficient number of span lengths
        lengths = poisson(lam=self.poisson_lambda, size=(num_to_mask,))
        while np.cumsum(lengths, 0)[-1] < num_to_mask:
            lengths = np.concatenate([lengths, poisson(lam=self.poisson_lambda, size=(num_to_mask,))])

        # remove all spans of length 0
        # Note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        lengths = lengths[lengths > 0]

        # trim to about num_to_mask tokens
        idx = np.argmin(np.abs(np.cumsum(lengths, 0) - num_to_mask)) + 1
        lengths = lengths[: idx + 1]

        # select span start indices
        # print("IS TOKEN")
        # print(is_token)
        # print(sum(list(map(lambda x: 1 if(x) else 0, is_token[0]))))
        token_indices = np.argwhere(is_token == 1)
        # print("TOKEN INDICES")
        # print(token_indices)
        span_starts = permutation(token_indices.shape[0])[: lengths.shape[0]]

        # prepare mask
        masked_indices = np.array(token_indices[span_starts])
        # print("MASKED INDICES")
        # print(masked_indices)
        mask = np.full_like(labels, fill_value=False)

        # mask span start indices
        for mi in masked_indices:
            mask[tuple(mi)] = True
        lengths -= 1

        # fill up spans
        max_index = labels.shape[1] - 1
        remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)
        while np.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            lengths -= 1
            remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        mask[np.where(special_tokens_mask)] = False
        inputs[np.where(mask)] = self.mask_token_id

        if not do_permutate:
            labels[np.where(mask)] = -100
        else:
            labels[np.where(special_tokens_mask)] = -100

        # remove mask tokens that are not starts of spans
        to_remove = (mask == 1) & np.roll((mask == 1), 1, 1)
        new_inputs = np.full_like(labels, fill_value=self.pad_token_id)

        # splits = list(map(lambda x: x.reshape(-1),  np.split(inputs_copy, indices_or_sections=2, axis=0))
        for i, example in enumerate(np.split(inputs, indices_or_sections=new_inputs.shape[0], axis=0)):
            new_example = example[0][~to_remove[i]]
            new_inputs[i, 0 : new_example.shape[0]] = new_example

        # batching now fixed
        return new_inputs, labels
