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
from utils import SeqRecDataset

from transformers import BartConfig, BartForConditionalGeneration

from config import BARTforSeqRecConfig
from model  import BARTforSeqRec

def pretrain(
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


def main():
    device = 'cuda' if cuda.is_available() else 'cpu'

if __name__ == '__main__':
    main()