import sys
import gzip
import pandas as pd
from collections import defaultdict

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]
true = True
false = False
class PreProcessorAmazonReview():
    def __init__(
        self,
        dataframe = pd.DataFrame(),
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
        self.dataframe = dataframe
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
        
        self.dataframe = pd.DataFrame.from_dict(df, orient = 'index')
    
    def _dataframe_to_dict(self):
        self.data_dict = self.dataframe.to_dict(orient = "index")

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
        
        return clean_dataset

    def preprocess(
        self,
        use_pre_loaded_map = False,
        to_dataframe = True
    ):
        self._dataframe_to_dict()
        self._count()
        preprocessed_data = self._clean_dataset(use_pre_loaded_map)
        #print(preprocessed_data)

        if to_dataframe :
            return pd.DataFrame(preprocessed_data)
        else :
            return preprocessed_data

def main(input_path, output_path):
    ourPreProcessorAmazonReview = PreProcessorAmazonReview()
    ourPreProcessorAmazonReview.parse_from_path(input_path)
    print("Unprocessed")
    print(ourPreProcessorAmazonReview.dataframe.head(10))

    df_preproessed_data = ourPreProcessorAmazonReview.preprocess(use_pre_loaded_map = False, to_dataframe= True)
    print("Processed")
    print(df_preproessed_data.head(10))
    
    print("Ordered")
    df_preproessed_data = df_preproessed_data.sort_values(['user_id', 'time'], ascending=[True, True], ignore_index=True)
    print(df_preproessed_data.head(10))
    df_preproessed_data.to_csv(output_path)

    
if __name__ == '__main__':
    main(input_file_path, output_file_path)