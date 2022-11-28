import numpy as np
import torch
import random
import copy
import wandb
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from utils import SeqRecDataset

def train(
    epoch,
    model,
    device,
    loader,
    optimizer
):
    #model.train()
    
    for _,data in enumerate(loader, 0):
        #optimizer.zero_grad()
        
        user_ids     = data['user_id']
        input_ids    = data['input_ids']
        positive_ids = data['positive_ids'] 
        negative_ids = data['negative_ids'] 
        
        print(user_ids, input_ids, positive_ids, negative_ids)
        
        ##TODO:
        ## ADD Model updating
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
        #outputs = model(input_ids = ids, attention_mask = mask, labels=y)
        
        #loss = outputs[0]
        #logits = outputs.logits
        #pred  = torch.argmax(F.softmax(logits,dim=1),dim=1)
        
        #correct = pred.eq(y)
        #total_correct += correct.sum().item()
        #total_len += len(labels)
        #loss.backward()
        #optimizer.step()


def valid(
    epoch,
    model,
    device,
    loader
):  
    #model.eval()
    for _,data in enumerate(loader, 0):
        #optimizer.zero_grad()
        
        user_ids     = data['user_id']
        input_ids    = data['input_ids']
        positive_ids = data['positive_ids'] 
        negative_ids = data['negative_ids'] 
        
        print(user_ids, input_ids, positive_ids, negative_ids)

        ##TODO:
        ## ADD Model validation
        

def main():
    wandb.init(project="BART SeqRec results")
    
    config = wandb.config           # Initialize config
    config.TRAIN_BATCH_SIZE = 2    # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 1     # input batch size for testing (default: 1)
    config.TRAIN_EPOCHS =  1        # number of epochs to train (default: 10)
    config.VAL_EPOCHS = 1  
    config.LEARNING_RATE = 4.00e-05 # learning rate (default: 0.01)
    config.SEED = 420               # random seed (default: 42)
    config.MAX_LEN = 512

    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }

    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }
    
    torch.backends.cudnn.deterministic = True

    dfdataset  = pd.read_csv('data/beauty.csv')
    train_for_testing_dataset = SeqRecDataset(dfdataset, is_train = True, for_testing = True)
    test_for_testing_dataset  = SeqRecDataset(dfdataset, is_train = False, for_testing = True)

    print(len(train_for_testing_dataset.sequences),train_for_testing_dataset.sequences[0]["sequence"])
    print(len(test_for_testing_dataset.sequences), test_for_testing_dataset.sequences[0]["sequence"])
    #train_for_validation_dataset = SeqRecDataset(dfdataset, is_train = True, for_testing = False)
    #test_for_validation_dataset  = SeqRecDataset(dfdataset, is_train = False, for_testing = False)

    train_for_testing_set_loader = DataLoader(train_for_testing_dataset, **train_params)
    test_for_testing_set_loader  = DataLoader(test_for_testing_dataset, **val_params)

    for epoch in range(config.TRAIN_EPOCHS):
        train(None,None,None,train_for_testing_set_loader,None)

    #for epoch in range(config.VAL_EPOCHS):
    #    train(None,None,None,test_for_testing_set_loader,None)


if __name__ == '__main__':
    main()