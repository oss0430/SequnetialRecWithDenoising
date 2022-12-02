import numpy as np
from numpy.random import permutation, poisson
import torch
import random
import copy
import math
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import math
from dataclasses import dataclass
from typing import Dict, List, Optional
from dataclasses import dataclass
import pdb

class SeqRecDataset(Dataset):
    def __init__(
        self, 
        dataframe, 
        is_train = True,
        for_testing = True,
        max_len = 384
    ):
        self.data          = dataframe
        self.review_user   = dataframe['user_id']
        self.review_item   = dataframe['item_id']
        self.is_train      = is_train
        self.for_testing   = for_testing

        final_idx = -1
        if for_testing:
            final_idx = -1
        else: 
            final_idx = -2

        if is_train :
            final_idx = final_idx - 1
        """
        Original Sequence 
        [1] - [2] - [3] - [4] - [5]

        training for testing
            Sequence = [1] - [2] - [3] - [4] 
        training for validation
            Sequence = [1] - [2] - [3]

        testing for testing
            Sequence = [1] - [2] - [3] - [4] - [5]
        testing for validation
            Sequence = [1] - [2] - [3] - [4]
        """
        sequences = {}
        """
        sequences = {   1 :{
                            "user_id" : 0
                            "sequence": [14, 18, 800, 12]
                            },
                        2 : {
                            "user_id" : 1
                            "sequence": [12, 15, 188, 13, 250]
                        } 
                    }
        """
        usernum = 0
        itemnum = 0
        user_item_list = defaultdict(list)
        for idx, row in dataframe.iterrows():
            u = row['user_id']
            i = row['item_id']
        
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            user_item_list[u].append(i)

        idx = 0
        
        for user, item_sequence in user_item_list.items():
            final_positive_idx = len(item_sequence) + final_idx 
            #print(len(item_sequence), final_positive_idx)
            if final_positive_idx >= 0 : 
                sequences[idx] = {"user_id" : user, "sequence" : item_sequence[:final_positive_idx + 1]}
                idx = idx + 1

        self.usernum = usernum
        self.itemnum = itemnum
        self.sequences = sequences
        self.padding_idx = 0
        self.bos_idx = itemnum + 2
        self.eos_idx = itemnum + 3
        self.item_mask_index = itemnum + 1
        self.max_len = max_len


    def __len__(self):
        return len(self.sequences)


    def _sequence_noising(
        self,
        input_seq,
        mask_token_id,
        permutation_segment_token_id
    ):  
        ##TODO
        ## MAKE NOISING Function
        noised_ids = input_seq
        return noised_ids


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


    def _pad_and_trunc_by_max_len(
        self,
        sequence = []
    ):
        ## sequence = [1] - [2] - [3]
        ## max_len  = 5
        ## return   = [0] - [0] - [1] - [2] - [3] 
        new_sequence =[self.bos_idx] + sequence + [self.eos_idx] + [self.padding_idx] * (self.max_len - len(sequence) - 2)

        return new_sequence[:self.max_len]


    def _sample_from_training_set_by_index(
        self,
        index
    ):  
        ## Training Sequence will be used to train the model for sequencial recommendation
        ## if training sequence for user 1
        ## [1] - [3] - [4] - [8] - [11] 
        ## model will predict [11] from
        ## [1] - [3] - [4] - [8]
        
        ## input_sequence    = [1] - [3] - [4] - [8]
        ## positive_sequence = [1] - [3] - [4] - [8] - [11]
        ## negative_sequence = [77] - [123] - [2534] - [22] - [52]
        
        single_data = copy.deepcopy(self.sequences[index])
        ts = set(single_data["sequence"])

        user_id = single_data["user_id"]
        input_sequence    = single_data["sequence"][:-1]
        target_item       = single_data["sequence"][-1]
        positive_sequence = single_data["sequence"]
        negative_sequence = []
        for _ in positive_sequence:
            negative_sequence.append(self._random_neq(1, self.itemnum, ts))

        return user_id, input_sequence, positive_sequence, negative_sequence, target_item


    def __getitem__(
        self, 
        index
    ):
        user_id, input_sequence, positive_sequence, negative_sequence, target_item = self._sample_from_training_set_by_index(index)
        ## Add mask at the end of input sequence 
        input_sequence.append(self.item_mask_index)
        if len(input_sequence) > self.max_len:
            pdb.set_trace()

        user_id         = np.array([user_id])
        input_ids       = np.array(self._pad_and_trunc_by_max_len(input_sequence))
        positive_ids    = np.array(self._pad_and_trunc_by_max_len(positive_sequence))
        negative_ids    = np.array(self._pad_and_trunc_by_max_len(negative_sequence))
        target_item     = np.array([target_item])
        return {
            'user_id'        : user_id,
            'input_ids'      : input_ids,
            'positive_ids'   : positive_ids,
            'negative_ids'   : negative_ids,
            'target_item'    : target_item
        }


class SeqRecTrainer():
    ## TODO:
    ## Implement Metrics Class
    ## Currently unused
    def __init__(
        self,
        dataset,
        model
    ):
        self.datset = dataset
        self.model  = model

    def _ht(
        self,
        predictions
    ):
        ht = 0.0
        rank = predictions.argsort().argsort()[0].item()
        if rank < 10:
            ht += 1
        return ht

    def _ndcg(
        self,
        predictions
    ):
        ndcg = 0.0
        rank = predictions.argsort().argsort()[0].item()
        if rank < 10:
            ndcg += 1 / np.log2(rank + 2)
        return ndcg

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

    def _get_metrics(
        self,
        num_candidate = 10
    ):
        return {
            "ndcg" : self._ndcg(num_candidate),
            "ht"   : self._ht(num_candidate)
        }



        
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
        
    
    # mask_ratio: float = 0.5
    # poisson_lambda: float = 3.0
    # permutate_sentence_ratio: float = 0.0
    # eos_token_id : int = 57289 + 3
    # bos_token_id : int = 57289 + 2
    # pad_token_id : int = 0
    # mask_token_id : int = 57289 + 1
    # pad_to_multiple_of: int = 16


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
            print(batch['input_ids'])
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
