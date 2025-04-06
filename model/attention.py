import torch
from torch import tensor
import torch.nn as nn

class SelfAttention_V2(nn.Module):
    def __init__(self, embed_dim_in,embed_dim_out,bias=False):
        super(SelfAttention_V2,self).__init__()
        self.W_query = nn.Linear(embed_dim_in,embed_dim_out,bias=bias)
        self.W_key = nn.Linear(embed_dim_in,embed_dim_out,bias=bias)
        self.W_value = nn.Linear(embed_dim_in,embed_dim_out,bias=bias)

    def forward(self,x):
        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)

        attention_scores = query @ key.T
        attention_weights = (attention_scores / torch.sqrt(tensor(x.shape[-1]))).softmax(dim=-1)

        context_vectors = attention_weights @ value
        return context_vectors