import torch
import numpy as np
from transformers import BartForConditionalGeneration, BartModel
import pdb
class BARTforSeqRec(torch.nn.Module):
    def __init__(self, BARTforSeqRecConfig):
        super(BARTforSeqRec, self).__init__()
        self.config = BARTforSeqRecConfig
        self.BartForConditionalGeneration = BartForConditionalGeneration(BARTforSeqRecConfig)
        self.mask_token_id = BARTforSeqRecConfig.mask_token_id
        self.item_embedding = self.BartForConditionalGeneration.model.shared
        print("padding_idx:", self.item_embedding.padding_idx)
    
    def forward(
        self,
        input_ids = None,
        decoder_input_ids = None,
        labels = None,
        neg_seqs = None,
        user_ids = None
    ):
        return self.BartForConditionalGeneration.forward(input_ids = input_ids,
                decoder_input_ids = decoder_input_ids,
                labels = labels) 
    
    def predict(
        self,
        user_ids = None,
        input_ids = None,
        candidate_items = None,
        top_N = 10
    ):
        
        logits = self.BartForConditionalGeneration(input_ids = input_ids).logits
    
        masked_indexs = (input_ids == self.mask_token_id).nonzero()
        
        value_list = []
        predictions_list = []

        for indices in masked_indexs:
            probs = logits[indices[0], indices[1], :].softmax(dim=0)
            value, prediction  = probs.topk(top_N)
            value_list.append(value)
            predictions_list.append(prediction)
        
        return torch.cat(value_list), torch.cat(predictions_list)

    def new_predict(
        self,
        user_ids = None,
        input_ids = None,
        candidate_items = None,
        top_N = 10
    ):
        
        logits = self.BartForConditionalGeneration(input_ids = input_ids).logits
        
        #print(logits)
        #print(logits.shape)
        #masked_index = (input_ids[0] == self.mask_token_id).nonzero().item()
        probs = logits[:, -1].softmax(dim=0)
        #print(probs.shape)
        values, predictions = probs.topk(top_N)

        return values, predictions, logits


    def generate(
        self,
        user_ids = None,
        input_ids = None,
        candidate_items = None,
        top_N = 10
    ) :
        return self.BartForConditionalGeneration.generate(input_ids = input_ids, num_beams = top_N, early_stopping=True)


class BARTforSeqRecWithBaseBart(torch.nn.Module):
    def __init__(self, config):
        super(BARTforSeqRecWithBaseBart, self).__init__()
        self.config = config
        self.bartBase = BartModel(config)
        self.mask_token_id = config.mask_token_id
        self.item_embedding = self.bartBase.shared

    
    def forward(
        self,
        user_ids = None,
        input_ids = None,
        decoder_ids = None,
        labels = None
    ):
        return self.bartBase.forward(
                input_ids = input_ids,
                decoder_input_ids = decoder_ids,
        ) 
    
    def predict(
        self,
        user_ids = None,
        input_ids = None,
        candidate_items = None,
        top_N = 10
    ):
        
        logits = self.BartForConditionalGeneration(input_ids = input_ids).logits
        masked_index = (input_ids[0] == self.mask_token_id).nonzero().item()
        probs = logits[0, masked_index].softmax(dim=0)

        values, predictions = probs.topk(top_N)

        return values, predictions

