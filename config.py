import os
import pandas as pd
import argparse
import torch
from torch import cuda
from transformers import BartConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda:0") if cuda.is_available() else torch.device("cpu")

data_path = 'data/beauty.csv'
DFDATASET = pd.read_csv(data_path)

def get_args():
    parser = argparse.ArgumentParser(description='BART4Rec with Amazon Beauty')
    parser.add_argument('-f', default='', type=str)

    # Architecture
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.2)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--intermediate_size', type=int, default=256)
    parser.add_argument('--max_position_embeddings', type=int, default=200)
    parser.add_argument('--max_lengths', type=int, default=50)
    parser.add_argument('--num_encoder_attention_heads', type=int, default=2)
    parser.add_argument('--num_decoder_attention_heads', type=int, default=2)
    parser.add_argument('--num_encoder_layers', type=int, default=2)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--initializer_range', type=float, default=0.02)
    
    # Training Setting
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--valid_batch_size', type=int, default=1)
    parser.add_argument('--clip', type=float, default=0.8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--valid_num_epochs', type=int, default=1)
    parser.add_argument('--when', type=int, default=20,
                        help='when to decay learning rate (default: 20)')
    parser.add_argument('--patience', type=int, default=20,
                        help='when to stop training if best never change')

    # Pretraining Setting
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--poisson_lambda', type=float, default=3.0)
    parser.add_argument('--permutate_sentence_ratio', type=float, default=0.0)

    # Logistics
    parser.add_argument('--seed', type=int, default=420)

    args = parser.parse_args()
    return args

class BARTforSeqRecConfig(BartConfig):
    def __init__(
        self,
        **common_kwargs
    ):
        super().__init__(
            **common_kwargs
        )
