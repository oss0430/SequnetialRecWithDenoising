import torch
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

        self.sequence_train = sequence_train
        self.sequence_test  = sequence_test

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


    def __getitem__(
        self, 
        index
    ):
        ## TODO
        ## Find out BART style input and output
        input_ids      = torch.tensor(self.sequence_train[index])
        noised_ids     = self._sequence_noising(input_ids)
        labels_y       = torch.tensor(self.sequence_test[index])
        
        return {
            'input_text_ids' : input_ids.to(dtype=torch.long),
            'labels_y'       : labels_y.to(dtype=torch.long)
        }

