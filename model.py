import torch
import numpy as np
from transformers import BartForConditionalGeneration
import pdb
class BARTforSeqRec(torch.nn.Module):
    def __init__(self, BARTforSeqRecConfig):
        super(BARTforSeqRec, self).__init__()
        self.config = BARTforSeqRecConfig
        self.BartForConditionalGeneration = BartForConditionalGeneration(BARTforSeqRecConfig)
        self.mask_token_id = BARTforSeqRecConfig.mask_token_id
    
    def forward(
        self,
        input_ids = None,
        decoder_ids = None,
        labels = None,
        neg_seqs = None,
        user_ids = None
    ):
        pdb.set_trace()
        return self.BartForConditionalGeneration.forward(input_ids = input_ids,
                decoder_input_ids = decoder_ids,
                labels = labels) 
    
    def predict(
        self,
        user_ids,
        seqs,
        candidate_items = None,
        top_N = 10
    ):
        logits = self.BartForConditionalGeneration(seqs).logits

        masked_index = (seqs[0] == self.mask_token_id).nonzero().item()
        probs = logits[0, masked_index].softmax(dim=0)

        values, predictions = probs.topk(top_N)

        return values, predictions
