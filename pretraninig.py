##TODO:
## Implement Pretraining Mechanism
## (for denoising)
## Implement Loading from Pretrained Model (at traning.py)

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

from config import *
from model  import BARTforSeqRec

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
    print("Start model pretraining....")

    total_loss = 0
    index = 0

    print("Pretraining Start !")
    for _,data in enumerate(loader, 0):
        # user_ids     = data['user_id'].to(device)
        input_ids    = torch.tensor(data['input_ids']).to(device, dtype = torch.long) #item_seq
        decoder_ids  = torch.tensor(data['decoder_input_ids']).to(device, dtype = torch.long) #decoder_seq
        labels       = torch.tensor(data['labels']).to(device, dtype = torch.long) #mask item labels

        outputs = model.forward(input_ids = input_ids, decoder_ids = decoder_ids, labels = labels)

        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if _%10 == 0:
            wandb.log({"Training Loss": loss.item()})
        
        if _%500==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        index += 1

    print(f"Epoch: {epoch}, Loss: {(total_loss / index)}")
    return total_loss / index


def main():
    args = get_args()
    device = DEVICE

    wandb.init(project="BART SeqRec Pretraining results")
    wandb.config.update(args)

    set_seed(args.seed)

    train_params = {
        'batch_size': args.valid_batch_size,
        'shuffle': False,
        'num_workers': 0
        }

    print("Start loading the data....")
    train_for_testing_dataset = SeqRecDataset(DFDATASET, is_train = True,  for_testing = True, max_len=args.max_lengths)

    itemnum = train_for_testing_dataset.itemnum

    data_collator = DataCollatorForDenoisingTasks(mask_ratio = args.mask_ratio, poisson_lambda = args.poisson_lambda, \
         permutate_sentence_ratio = args.permutate_sentence_ratio, eos_token_id = itemnum + 3, bos_token_id = itemnum + 2, \
            pad_token_id = 0, mask_token_id = itemnum + 1, pad_to_multiple_of = 16)
    train_for_testing_loader_with_noise = DataLoader(train_for_testing_dataset, collate_fn=data_collator, **train_params)

    ## 0 : padding_token_id
    ## itemnum + 1 : mask_token_id
    ## itemnum + 2 : bos_token 
    ## itemnum + 3 : eos_token 
    item_embedding_size = itemnum + 5

    modelConfig = BARTforSeqRecConfig(vocab_size = item_embedding_size, \
        pad_token_id=0, bos_token_id= itemnum + 2, eos_token_id= itemnum + 3, mask_token_id= itemnum + 1, \
            d_model = args.hidden_size, encoder_layers = args.num_encoder_layers, decoder_layers = args.num_decoder_layers, \
                encoder_attention_heads = args.num_encoder_attention_heads, decoder_attention_heads = args.num_decoder_attention_heads, \
                    decoder_ffn_dim = args.intermediate_size, encoder_ffn_dim = args.intermediate_size, max_position_embeddings= args.max_position_embeddings)
    
    model = BARTforSeqRec(modelConfig)
    model.to(device)

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=args.learning_rate)

    best_train_loss = 1000

    for epoch in range(args.train_num_epochs):
        train_loss = train(epoch + 1, model, device, train_for_testing_loader_with_noise, optimizer)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'pretrained_models/model.pt')
            print("Pretrained model updated !")


if __name__ == '__main__':
    main()
