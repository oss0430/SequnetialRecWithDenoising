import torch
import numpy as np
from transformers import BartEncoder, BartDecoder


class RecwithSequenceDenosingEmbeddingLayer(torch.nn.Module):
    def __init__(
        self,
        user_number,
        item_number,
        embedding_size
    ):
        self.user_embedding = torch.nn.Embedding(num_embeddings = user_number, embedding_dim=embedding_size)
        self.item_embedding = torch.nn.Embedding(num_embeddings = item_number, embedding_dim=embedding_size)

    def forward(
        self,
        user_ids,
        item_ids
    ):
        ##
        sequence = np.zeros(12)
        return sequence

class RecWithSequenceDenoising(torch.nn.Module):
    def __init__(
        self
    ):
        self.encoder = BartEncoder
        self.decoder = BartDecoder

