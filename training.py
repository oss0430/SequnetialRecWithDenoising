import numpy as np
import torch
import random
import copy
import wandb
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch import cuda
from utils import SeqRecDataset

from transformers import BartConfig, BartForConditionalGeneration

from config import BARTforSeqRecConfig
from model  import BARTforSeqRec

def train(
    epoch,
    model,
    device,
    loader,
    optimizer
):
    model.train()
    
    for _,data in enumerate(loader, 0):
        user_ids     = data['user_id'].to(device)
        input_ids    = data['input_ids'].to(device)
        positive_ids = data['positive_ids'].to(device)
        negative_ids = data['negative_ids'].to(device)
        
        outputs = model.forward(user_ids = user_ids, seqs = input_ids, pos_seqs = positive_ids, neg_seqs = negative_ids)
        #print(outputs)
        loss = outputs[0]
        #logits = outputs.logits
        #pred  = torch.argmax(F.softmax(logits,dim=1),dim=1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #correct = pred.eq(y)
        #total_correct += correct.sum().item()
        #total_len += len(labels)

        if _%10 == 0:
            wandb.log({"Training Loss": loss.item()})
        
        if _%500==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

    return loss.item()


def valid(
    epoch,
    model,
    device,
    loader
):  
    model.eval()
    
    ht   = np.array([0.0])
    ndcg = np.array([0.0])
    user_numbers = len(loader)

    for _,data in enumerate(loader, 0):
        #optimizer.zero_grad()
        
        user_ids     = data['user_id'].to(device)
        input_ids    = data['input_ids'].to(device)
        positive_ids = data['positive_ids'].to(device)
        negative_ids = data['negative_ids'].to(device)
        target_item  = data['target_item'].to(device)
        
        values, predictions = model.predict(user_ids, input_ids, candidate_items = None, top_N = 10)
        
        rank = (predictions == target_item).nonzero(as_tuple=True)[0].to(torch.device("cpu")).numpy()
        if len(rank) > 0:
            #print(predictions, target_item, rank)
            ht += 1
            ndcg += np.log2(rank + 2)
        else :
            #print(predictions, target_item, rank)
            ht += 0
            ndcg += 0
    
    ht = ht / user_numbers
    ndcg = ndcg / user_numbers
    print("ht :" ,ht, " ndcg : ", ndcg)
    return ht, ndcg

def main():
    device = 'cuda' if cuda.is_available() else 'cpu'
    wandb.init(project="BART SeqRec results")
    
    config = wandb.config           # Initialize config
    config.TRAIN_BATCH_SIZE = 2    # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 1     # input batch size for testing (default: 1)
    config.TRAIN_EPOCHS =  1        # number of epochs to train (default: 10)
    config.VAL_EPOCHS = 1  
    config.LEARNING_RATE = 4.00e-05 # learning rate (default: 0.01)
    config.SEED = 420               # random seed (default: 42)
    config.MAX_LEN = 50

    config.HIDDEN_DIM = [16, 32, 64, 128, 256, 512, 1024]  # hiddem dimension size (default: 1024)
    config.LAYER_NUM = 2                                   # number of layers (default: 12)
    config.HEAD_NUM = 2                                    # number of attention heads for each attention layer (default: 16)
    config.FFN_DIM = 32                                    # the dimensionality of each head (default: 4096)

    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }

    ## Validation Batch size must be 1
    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }
    
    torch.backends.cudnn.deterministic = True

    dfdataset  = pd.read_csv('data/beauty.csv')
    
    test_for_testing_dataset  = SeqRecDataset(dfdataset, is_train = False, for_testing = True, max_len= config.MAX_LEN)
    train_for_testing_dataset = SeqRecDataset(dfdataset, is_train = True,  for_testing = True, max_len= config.MAX_LEN)

    itemnum = test_for_testing_dataset.itemnum
    usernum = test_for_testing_dataset.usernum

    print(len(train_for_testing_dataset.sequences),train_for_testing_dataset.sequences[0]["sequence"])
    print(len(test_for_testing_dataset.sequences), test_for_testing_dataset.sequences[0]["sequence"])
    #train_for_validation_dataset = SeqRecDataset(dfdataset, is_train = True, for_testing = False)
    #test_for_validation_dataset  = SeqRecDataset(dfdataset, is_train = False, for_testing = False)

    train_for_testing_set_loader = DataLoader(train_for_testing_dataset, **train_params)
    test_for_testing_set_loader  = DataLoader(test_for_testing_dataset, **val_params)

    ## 0 : padding_token_id
    ## itemnum + 1 : mask_token_id
    ## itemnum + 2 : bos_token (not used)
    ## itemnum + 3 : eos_token (not used)
    item_embedding_size = itemnum + 4

    ## TODO:
    ## Need to Match Parameters Number
    ## Currently layer numbers are 12 for encoder 16 for decoder
    ## While BERT4Recs are 2 layer
    modelConfig = BARTforSeqRecConfig(vocab_size = item_embedding_size, \
        pad_token_id=0, bos_token_id= itemnum + 2, eos_token_id= + 3, mask_token_id= itemnum + 1, \
            d_model = config.HIDDEN_DIM[0], encoder_layers = config.LAYER_NUM, decoder_layers = config.LAYER_NUM, \
                encoder_attention_heads = config.HEAD_NUM, decoder_attention_heads = config.HEAD_NUM, decoder_ffn_dim = config.FFN_DIM, \
                    encoder_ffn_dim = config.FFN_DIM, max_position_embeddings= config.MAX_LEN)
    model = BARTforSeqRec(modelConfig)
    model.to(device)

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE)

    best_train_loss = 1000
    best_valid_ht, best_valid_ndcg = 0.0, 0.0

    for epoch in range(config.TRAIN_EPOCHS):
        train_loss = train(epoch + 1, model, device, train_for_testing_set_loader, optimizer)

        # Save the best training model
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'trained_models/model.pt')

    for epoch in range(config.VAL_EPOCHS):
        model.load_state_dict(torch.load('trained_models/model.pt'))
        hitratio, ndcg = valid(epoch + 1, model, device, test_for_testing_set_loader)

        # Save the model validation results
        if hitratio > best_valid_ht and ndcg > best_valid_ndcg:
            best_valid_ht = hitratio
            best_valid_ndcg = ndcg
        
        wandb.log({"Best Valid HitRatio": best_valid_ht, "Best Valid NDCG": best_valid_ndcg})
    
    print("=======================================")
    print("Best Model Result")
    print("Epoch: {best_epoch}, Loss: {best_train_loss}, HitRatio: {best_valid_ht}, NDCG: {best_valid_ndcg}")
    print("=======================================")

    ## TODO:
    ##  Add saving model parameters functions 
    ##  Add saving result functions

if __name__ == '__main__':
    main()