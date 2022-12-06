import numpy as np
import torch
import random
import copy
import wandb
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch import cuda
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import SeqRecDataset, DataCollatorForDenoisingTasks

from transformers import BartConfig, BartForConditionalGeneration
from torchmetrics import RetrievalHitRate
from torchmetrics.functional import retrieval_hit_rate

from transformers import Trainer
from config import *
from model  import BARTforSeqRec, BARTforSeqRecWithBaseBart
import pdb
import time
import datetime
import tqdm

class Solver(object):
    def __init__(
        self,
        hyp_params,
        train_loader,
        valid_loader,
        test_loader,
        is_train=True,
        model=None,
        pretrained_model=None
    ):
        self.args = args = hyp_params
        self.epoch_i = 0
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.is_train = is_train
        self.model = model

        ## 0 : padding_token_id
        ## itemnum + 1 : mask_token_id
        ## itemnum + 2 : bos_token 
        ## itemnum + 3 : eos_token 
        item_embedding_size = self.args.itemnum + 5
        mask_token_id = self.args.itemnum + 1
        bos_token_id = self.args.itemnum + 2
        eos_token_id = self.args.itemnum + 3

        # BART model configuration
        self.model_config = BARTforSeqRecConfig(vocab_size = item_embedding_size, \
            pad_token_id=0, bos_token_id=bos_token_id, eos_token_id=eos_token_id, mask_token_id=mask_token_id, \
                d_model = args.hidden_size, encoder_layers = args.num_encoder_layers, decoder_layers = args.num_decoder_layers, \
                    encoder_attention_heads = args.num_encoder_attention_heads, decoder_attention_heads = args.num_decoder_attention_heads, \
                        decoder_ffn_dim = args.intermediate_size, encoder_ffn_dim = args.intermediate_size, max_position_embeddings= args.max_position_embeddings, \
                            dropout=args.dropout, attention_dropout=args.attention_probs_dropout_prob, init_std=args.initializer_range)

        # Initialize the model
        if model is None:
            self.model = model = BARTforSeqRec(self.model_config)
        
        if torch.cuda.is_available():
            self.args.device = self.device = torch.device("cuda")
            model = model.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.optimizer = getattr(torch.optim, args.optim)(
            self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=args.when, factor=0.5, verbose=True)

    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################

    def train_and_eval(self):
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        
        print(self.device)

        def train(model, optimizer):
            epoch_loss = 0
            proc_size = 0

            model.train()
            num_batches = self.args.n_train // self.args.train_batch_size
            start_time = time.time()
            print(self.device)
            for i_batch, data in enumerate(self.train_loader, 0):
                #  user_ids     = data['user_id'].to(device) #user_id
                input_ids    = torch.tensor(data['input_ids']).to(self.device, dtype = torch.long) #item_seq
                #  target_item    = torch.tensor(data['target_item']).to(device, dtype = torch.long)
                #  decoder_ids  = torch.tensor(data['decoder_input_ids']).to(device, dtype = torch.long) #decoder_seq
                #  labels       = torch.tensor(data['labels']).to(device, dtype = torch.long) #mask item labels
                positive_ids = data['positive_ids'].to(self.device)
                #  negative_ids = data['negative_ids'].to(device)
                #print(input_ids, positive_ids)

                batch_size = input_ids.size(0)

                outputs = model.forward(input_ids = input_ids, labels = positive_ids)

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
                epoch_loss += loss.item() * batch_size
                proc_size += batch_size
                avg_loss = epoch_loss / proc_size
                elapsed_time = time.time() - start_time

                if i_batch%10 == 0:
                    wandb.log({"Training Loss": loss.item()})
                
                if i_batch%10==0:
                    print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                        format(epoch, i_batch, proc_size, elapsed_time * 1000, avg_loss))
                    print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
            return avg_loss
    
        def evaluate(model, test=False, top_N=10):
            model.eval()
            loader = self.test_loader if test else self.valid_loader
            total_loss = 0.0

            results = []
            truths = []
            hit_rate = 0.0

            with torch.no_grad():
                for _, data in enumerate(loader, 0):
                    user_ids     = data['user_id'].to(device)
                    input_ids    = data['input_ids'].to(device)
                    positive_ids = data['positive_ids'].to(self.device)
                    target_item  = data['target_item'].to(device)
                    target_item_value = target_item[:,0].view([-1,1])
                    #print(target_item_value)
                    #print(input_ids)
                    
                    if len(input_ids.nonzero()) == 0:
                        continue

                    device = self.device
                    batch_size = input_ids.size(0)

                    # values, predictions = model.predict(input_ids = input_ids, candidate_items = None, top_N = 10)
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits
                    probs = logits[:, -1].softmax(dim=0)
                    values, predictions = probs.topk(top_N)

                    total_loss += outputs[0]

                    # Compute hit ratio
                    hit_rate += retrieval_hit_rate(predictions, target_item_value, k=top_N)

                    # Collect the results into ntest if test else self.args.n_valid)
                    # results.append(predictions)
                    # truths.append(target_item_value)

            
            ht_average = hit_rate / (self.args.n_test if test else self.args.n_valid)
            # ndcg_average = ndcg / (self.args.n_test if test else self.args.n_valid)

            avg_loss = total_loss / (self.args.n_test if test else self.args.n_valid)
            # results = torch.cat(results)
            # truths = torch.cat(truths)
            return avg_loss, ht_average
    
        best_valid_loss = 1e8
        best_test_loss = 1e8
        patience = self.args.patience
        total_start = time.time()

        for epoch in range(1, self.args.num_epochs+1):
            start = time.time()
            self.epoch = epoch

            train_loss = train(model, optimizer)

            val_loss, _ = evaluate(model, test=False, topN=10)
            test_loss, hit_rate = evaluate(model, test=True, top_N=10)

            end = time.time()
            duration = end-start
            scheduler.step(val_loss)

            # Validation Loss
            print("-"*50)
            print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
            print("Result on Epoch {:2d}  | Hit Rate: {:5.4f}".format(epoch, hit_rate))
            print("-"*50)

            if val_loss < best_valid_loss:
                torch.save(model.state_dict(), 'trained_models/model.pt')

                # update best validation
                patience = self.args.patience
                best_valid_loss = val_loss

                if test_loss < best_test_loss:
                    best_epoch = epoch
                    best_test_loss = test_loss
                    best_test_ht = hit_rate

                    torch.save(model.state_dict(), 'trained_models/model.pt')
            
            else:
                patience -= 1
                if patience == 0:
                    break

            wandb.log(
                (
                    {
                        "Training Loss": train_loss,
                        "Validation Loss": val_loss,
                        "Test Loss": test_loss,
                        "Best Valid HitRatio": hit_rate,
                        "best_valid_loss": best_valid_loss,
                        "best_test_loss": best_test_loss
                    }
                )
            )

            print(f'Best epoch: {best_epoch}')
            print("Result on Epoch {:2d}  | Hit Rate: {:5.4f}".format(best_epoch, best_test_ht))

            total_end = time.time()
            total_duration = total_end - total_start
            print(f"Total training time: {total_duration}s, {datetime.timedelta(seconds=total_duration)}") 

