import numpy as np
import torch
import random
import copy
import wandb
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch import cuda
from utils import SeqRecDataset, DataCollatorForDenoisingTasks

from transformers import BartConfig, BartForConditionalGeneration

from transformers import Trainer
from config import *
from model  import BARTforSeqRec, BARTforSeqRecWithBaseBart
import pdb

def set_seed(seed):
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(DEVICE)
        torch.cuda.manual_seed_all(seed)
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        use_cuda = True 


def train(
    epoch,
    model,
    device,
    loader,
    optimizer
):
    model.train()
    print("Start model Training....")

    total_loss = 0

    #print(next(loader))

    for _,data in enumerate(loader, 0):
        #  user_ids     = data['user_id'].to(device) #user_id
        input_ids    = torch.tensor(data['input_ids']).to(device, dtype = torch.long) #item_seq
        #  target_item    = torch.tensor(data['target_item']).to(device, dtype = torch.long)
        #  decoder_ids  = torch.tensor(data['decoder_input_ids']).to(device, dtype = torch.long) #decoder_seq
        #  labels       = torch.tensor(data['labels']).to(device, dtype = torch.long) #mask item labels
        positive_ids = data['positive_ids'].to(device)
        #  negative_ids = data['negative_ids'].to(device)
        #print(input_ids, positive_ids)
        outputs = model.forward(input_ids = input_ids, decoder_input_ids = None, labels = positive_ids)

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
        total_loss += loss.item()
        if _%10 == 0:
            wandb.log({"Training Loss": loss.item()})
        
        if _%500==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')


    print(f"traning end , Epoch: {epoch}, Loss: {(total_loss / len(loader))}")
    return total_loss / len(loader)


def valid(
    epoch,
    model,
    device,
    loader,
    usernum
):  
    model.eval()
    print("Start model testing....")
    
    #print(next(loader))

    ht   = np.array([0.0])
    ndcg = np.array([0.0])
    user_numbers = usernum

    for _,data in enumerate(loader, 0):
        #optimizer.zero_grad()
        
        user_ids     = data['user_id'].to(device)
        input_ids    = data['input_ids'].to(device)
        target_item  = data['target_item'].to(device)
        target_item_value = target_item[:,0].view([-1,1])
        #print(target_item_value)
        #print(input_ids)
        
        if len(input_ids.nonzero()) == 0:
            continue
        #values, predictions, logits = model.new_predict(input_ids = input_ids, candidate_items = None, top_N = 10)
        
        values, predictions = model.predict(input_ids = input_ids, candidate_items = None, top_N = 10)
        
        nonzeros = (predictions == target_item_value).nonzero().to(torch.device("cpu")).numpy()
        
        ranks = [-1] * len(target_item_value)
        for nonzero_indices in nonzeros:
            ranks[nonzero_indices[0]] = nonzero_indices[1]
        
        #print(ranks)
        for rank in ranks:
            if rank == -1:
                #print(rank, values, predictions, target_item_value, "MISS")
                ht += 0
                ndcg += 0
            else:
                #print(rank, values, predictions, target_item_value, "HIT")
                ht += 1
                ndcg += 1/np.log2(rank + 2)
    
    ht_average = ht / user_numbers
    ndcg_average = ndcg / user_numbers
    print("ht :" ,ht, " ndcg : ", ndcg, "average : ",ht_average,ndcg_average, "user_number : ", user_numbers )
    return ht_average, ndcg_average

def main():
    args = get_args()

    wandb.init(project="BART SeqRec results")
    wandb.config.update(args)
    device = DEVICE

    wandb.config.PRETRAIN_EPOCHS = 0       # numbert of epochs to pretrain (default: 10)
    wandb.config.SEGMENT_SEQ = True       #for permutation segment in pretrain
    wandb.config.SEGMENT_LEN = 10         #segment length

    is_valid = False

    set_seed(args.seed)

    train_params = {
        'batch_size': args.train_batch_size,
        'shuffle': False,
        'num_workers': 0
        }

    ## Validation Batch size must be 1
    val_params = {
        'batch_size': args.valid_batch_size,
        'shuffle': False,
        'num_workers': 0
        }
    
    # torch.backends.cudnn.deterministic = True
    

    print("Start loading the data....")
    test_for_testing_dataset  = SeqRecDataset(DFDATASET, is_train = False, for_testing = True, max_len=args.max_lengths)
    train_for_testing_dataset = SeqRecDataset(DFDATASET, is_train = True,  for_testing = True, max_len=args.max_lengths)

    itemnum = test_for_testing_dataset.itemnum
    usernum = test_for_testing_dataset.usernum

    print(len(train_for_testing_dataset.sequences),train_for_testing_dataset.sequences[0]["sequence"])
    print(len(test_for_testing_dataset.sequences), test_for_testing_dataset.sequences[0]["sequence"])
    #train_for_validation_dataset = SeqRecDataset(dfdataset, is_train = True, for_testing = False)
    #test_for_validation_dataset  = SeqRecDataset(dfdataset, is_train = False, for_testing = False)
    train_for_testing_loader_with_noise = DataLoader(train_for_testing_dataset, **train_params)
    test_for_testing_loader  = DataLoader(test_for_testing_dataset,**val_params)


    ## 0 : padding_token_id
    ## itemnum + 1 : mask_token_id
    ## itemnum + 2 : bos_token 
    ## itemnum + 3 : eos_token 
    item_embedding_size = itemnum + 5

    # BART model configuration
    modelConfig = BARTforSeqRecConfig(vocab_size = item_embedding_size, \
        pad_token_id=0, bos_token_id= itemnum + 2, eos_token_id= itemnum + 3, mask_token_id= itemnum + 1, \
            d_model = args.hidden_size, encoder_layers = args.num_encoder_layers, decoder_layers = args.num_decoder_layers, \
                encoder_attention_heads = args.num_encoder_attention_heads, decoder_attention_heads = args.num_decoder_attention_heads, \
                    decoder_ffn_dim = args.intermediate_size, encoder_ffn_dim = args.intermediate_size, max_position_embeddings= args.max_position_embeddings, \
                        dropout=args.dropout, attention_dropout=args.attention_probs_dropout_prob, init_std=args.initializer_range)
    
    model = BARTforSeqRec(modelConfig)
    model.to(device)

    # Pretrained model load
    # # model.load_state_dict(torch.load('pretrained_models/model.pt'))

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=args.learning_rate)

    best_train_loss = 1000
    best_valid_ht, best_valid_ndcg = 0.0, 0.0
    
    if not is_valid:

        for epoch in range(args.num_epochs):
            train_loss = train(epoch + 1, model, device, train_for_testing_loader_with_noise, optimizer)


            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_epoch = epoch

        torch.save(model.state_dict(), 'trained_models/model.pt')
        print("saving at [trained_models/model.pt]")

    print("loading at [trained_models/model.pt]")
    model.load_state_dict(torch.load('trained_models/model.pt'))
    
    for epoch in range(args.valid_num_epochs):
        
        hitratio, ndcg = valid(epoch + 1, model, device, test_for_testing_loader, usernum)


        if hitratio > best_valid_ht and ndcg > best_valid_ndcg:
            best_valid_ht = hitratio
            best_valid_ndcg = ndcg

        wandb.log({"Best Valid HitRatio": best_valid_ht, "Best Valid NDCG": best_valid_ndcg})
    
    print("=======================================")
    print("Best Model Result")
    print("Epoch: ", best_epoch, ", Loss: ", best_train_loss, ", HitRatio: ", best_valid_ht, ", NDCG: ", best_valid_ndcg)
    print("=======================================")
    ## TODO:
    ##  Add saving model parameters functions 
    ##  Add saving result functions

if __name__ == '__main__':
    main()
