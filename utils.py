import numpy as np
import torch
import random
import copy
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

class RecommendationDataset(Dataset):
    def __init__(
        self, 
        dataframe, 
        is_valid = False
    ): 
        self.data          = dataframe
        self.review_user   = dataframe['user_id']
        self.review_item   = dataframe['item_id']
        self.is_valid      = is_valid
        user_item_list = defaultdict(list)
        
        if is_valid:
            final_idx = -2
        else: 
            final_idx = -1

        sequence_train = {}
        sequence_test  = {}
        """
        sequence_train ={   1 :{
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
        for idx, row in dataframe.iterrows():
            u = row['user_id']
            i = row['item_id']
        
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            user_item_list[u].append(i)

        idx = 0
        for user in user_item_list:
            nfeedback = len(user_item_list[user])

            if nfeedback < 2:
                sequence_train[idx] = {"user_id" : user, "sequence" : user_item_list[user]}
                sequence_test[idx]  = {"user_id" : user, "sequence" : []}
                
            else:
                sequence_train[idx] = {"user_id": user, "sequence" : user_item_list[user][:final_idx]}
                sequence_test[idx]  = {"user_id": user, "sequence" : [user_item_list[user][final_idx]]}
            
            idx = idx + 1

        self.usernum = usernum
        self.itemnum = itemnum
        self.sequence_train = sequence_train
        self.sequence_test  = sequence_test
        self.item_mask_index = itemnum + 1

    def __len__(self):
        return len(self.data)

    def _sequence_noising(
        self,
        ids
    ):  
        ##TODO
        ## MAKE NOISING Function
        noised_ids = ids
        return noised_ids

    
    def _random_neq(l, r, s):
        t = random.randint(l, r)
        while t in s:
            ## if random negative item is in set S
            ## Generate a new one
            t = random.randint(l, r)
        return t

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
        
        single_data = copy.deepcopy(self.sequence_train[index])
        ts = set(single_data[index]["sequence"])

        user_id = single_data["user_id"]
        input_sequence    = single_data["sequence"][:-1]
        positive_sequence = single_data["sequence"]
        negative_sequence = []
        for _ in positive_sequence:
            negative_sequence.append(self._random_neq(1, self.itemnum, ts))

        return user_id, input_sequence, positive_sequence, negative_sequence

    def __getitem__(
        self, 
        index
    ):
        user_id, input_sequence, positive_sequence, negative_sequence = self._sample_from_training_set_by_index(index)
        
        noised_ids      = self._sequence_noising(input_sequence)

        ## Add mask at the end of input sequence
        user_id         = torch.tensor([user_id])
        input_ids       = torch.tensor(input_sequence.append(self.item_mask_index))
        positive_ids    = torch.tensor(positive_sequence)
        negative_ids    = torch.tensor(negative_sequence)

        return {
            'user_id'        : user_id.to(dtype = torch.long),
            'input_ids'      : input_ids.to(dtype = torch.long),
            'positive_ids'   : positive_ids.to(dtype = torch.long),
            'negative_ids'   : negative_ids.to(dtype = torch.long)
        }
"""
    Mask filling example:
    ```python
    from transformers import BartTokenizer, BartForConditionalGeneration
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    TXT = "My friends are <mask> but they eat too many carbs."
    input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
    logits = model(input_ids).logits
    print(logits.shape) ## [1, 13, 50265]
    masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    probs = logits[0, masked_index].softmax(dim=0)
    values, predictions = probs.topk(5)
    tokenizer.decode(predictions).split()
    ['not', 'good', 'healthy', 'great', 'very']
    ```
"""

class SeqRecTrainer():

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
            HT += 1
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

    def _random_neq(l, r, s):
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
        
        single_data = self.dataset.sequence_train[index]
        ts = set(single_data[index]["sequence"])

        user_id = single_data["user_id"]
        input_sequence    = single_data["sequence"][:-1]
        positive_sequence = single_data["sequence"]
        negative_sequence = []
        for _ in positive_sequence:
            negative_sequence.append(self._random_neq(1, self.dataset.itemnum, ts))

        return user_id, input_sequence, positive_sequence, negative_sequence
