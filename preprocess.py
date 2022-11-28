import os
import sys
import random
import gzip
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

class PreProcessorAmazonReview():
    def __init__(
        self,
        data = pd.DataFrame(),
        user_map = {},
        item_map = {},
        user_count = {},
        item_count = {}
    ):
        """
        data = dataframe of user-item-timestamp
        user_map = dictionary of { user-id (simplyfied) : original-user-id }
        item_map = dictionary of { item-id (simplyfied) : original-item-id }
        user_count = dictionary of {user-id (original-user-id) : integer-value-of-user-frequency}
        item_count = dictionary of {item-id (original-user-id) : integer-value-of-itme-frequency}
        """
        self.data = data
        self.user_map = user_map
        self.item_map = item_map
        self.user_count = user_count
        self.item_count = item_count
    
    def _parse(
        self,
        path
    ):
        g = gzip.open(path, 'r')
        for l in g:
            yield eval(l)

    def parse_from_path(
        self,
        path
    ):
        i = 0
        df = {}
        for d in self._parse(path):
            df[i] = d
        i += 1
        
        self.data = pd.DataFrame.from_dict(df, orient = 'index')

    def _count(self):
        countU = defaultdict(lambda: 0)
        countI = defaultdict(lambda: 0)
        
        for idx, review in self.data_dict.items():
            raw_item_id = review['asin']
            rev_user_id = review['reviewerID']
            countU[rev_user_id]  += 1
            countI[raw_item_id] += 1

        self.user_count = countU
        self.item_count = countI


    def _clean_dataset(
        self,
        use_preloaded_user_item_maps = False
    ):
        if use_preloaded_user_item_maps:
            user_map = self.user_map
            item_map = self.item_map
        else :
            user_map = {}
            item_map = {}
        
        clean_dataset = {
            'user_id' : [],
            'item_id' : [],
            'time' : []
        }
        usernum = 0
        itemnum = 0
        for idx, review in self.data_dict.items():
            raw_item_id = review['asin']
            raw_user_id = review['reviewerID']
            time = review['unixReviewTime']
            
            if self.user_count[raw_user_id] < 5 or self.item_count[raw_item_id] < 5: #discard this interaction
                continue

            if raw_user_id in user_map: #already exist in the Map
                userid = user_map[raw_user_id]
    
            else: #non existing User in Map make one
                usernum += 1
                userid = usernum
                user_map[raw_user_id] = userid
        
            if raw_item_id in item_map: #already exist in the Map
                itemid = item_map[raw_item_id] 
    
            else: #non existing Item in Map make one
                itemnum += 1
                itemid = itemnum
                item_map[raw_item_id] = itemid
  
            clean_dataset['user_id'].append(userid)
            clean_dataset['item_id'].append(itemid)
            clean_dataset['time'].append(time)
        
        ##TODO:
        ## Add Sorting Function
        return clean_dataset


    def preprocess(
        self,
        use_pre_loaded_map = False
    ):
        self._dataframe_to_dict()
        self._count()
        preprocessed_data = self._clean_dataset(use_pre_loaded_map)

        return preprocessed_data
