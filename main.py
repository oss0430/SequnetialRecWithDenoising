import numpy as np
import torch
import random
import copy
import wandb
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch import cuda
# from utils import SeqRecDataset, DataCollatorForDenoisingTasks

from transformers import BartConfig, BartForConditionalGeneration

from transformers import Trainer
from config import *
from data_loader import get_loader
from model  import BARTforSeqRec, BARTforSeqRecWithBaseBart
from solver import Solver
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

def main():
    args = get_args()
    wandb.init(project="BART SeqRec results")
    wandb.config.update(args)
    device = DEVICE

    wandb.config.PRETRAIN_EPOCHS = 0       # numbert of epochs to pretrain (default: 10)
    wandb.config.SEGMENT_SEQ = True       #for permutation segment in pretrain
    wandb.config.SEGMENT_LEN = 10         #segment length

    set_seed(args.seed)

    # train_params = {
    #     'batch_size': args.train_batch_size,
    #     'shuffle': False,
    #     'num_workers': 0
    #     }

    # ## Validation Batch size must be 1
    # val_params = {
    #     'batch_size': args.valid_batch_size,
    #     'shuffle': False,
    #     'num_workers': 0
    #     }
    
    print("Start loading the data....")
    train_loader = get_loader(args, mode='train')
    print('Train set loaded !')
    valid_loader = get_loader(args, mode='valid')
    print('Valid set loaded !')
    test_loader = get_loader(args, mode='test')
    print('Test set loaded !')
    print('Finish loading the data....')

    solver = Solver(args, train_loader=train_loader, valid_loader=valid_loader, \
        test_loader=test_loader, is_train=True)

    solver.train_and_eval()

if __name__ == '__main__':
    main()