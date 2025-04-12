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
    
class CausalAttention(nn.Module):
    def __init__(self,embde_dim_in:int,
                 embed_dim_out:int,
                 context_length:int, # use for masking future tokens
                 dropout:float,
                 bias=False):
        super().__init__()
        self.W_query = nn.Linear(embde_dim_in,embed_dim_out,bias=bias)
        self.W_key = nn.Linear(embde_dim_in,embed_dim_out,bias=bias)
        self.W_value = nn.Linear(embde_dim_in,embed_dim_out,bias=bias)

        self.register_buffer("mask",
                             torch.triu(torch.ones(context_length,context_length),diagonal=1))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self,x:torch.Tensor):
        batch, num_tokens, embed_size = x.shape
        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)

        attention_scores = query @ key.transpose(1,2)

        attention_scores.masked_fill_(self.mask.bool()[:num_tokens,:num_tokens],
                                      -torch.inf)
        attention_scores = attention_scores / key.shape[-1]**0.5 # dividing by the square of the embedding size

        attention_weights = attention_scores.softmax(dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vectors = attention_weights @ value

        return context_vectors
    
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, embde_dim_in:int,
                 embed_dim_out:int,
                 context_length:int, # use for masking future tokens
                 dropout:float,
                 num_heads:int,
                 bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(embde_dim_in,embed_dim_out,context_length,dropout,bias) for _ in range(num_heads)
        ])

    def forrward(self,x):
        return torch.cat([
            head(x) for head in self.heads
        ],dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self,embde_dim_in:int,
                 embed_dim_out:int,
                 context_length:int, # use for masking future tokens
                 dropout:float,
                 num_heads:int,
                 bias=False):
        super(MultiHeadAttention,self).__init__()

        assert embed_dim_out % num_heads == 0 , "embed_dim_out must be divisible by num_heads"

        self.embed_dim_out = embed_dim_out
        self.num_heads = num_heads
        self.head_dim = embed_dim_out // num_heads

        self.W_query = nn.Linear(embde_dim_in,embed_dim_out,bias=bias)
        self.W_key = nn.Linear(embde_dim_in,embed_dim_out,bias=bias)
        self.W_value = nn.Linear(embde_dim_in,embed_dim_out,bias=bias)

        self.register_buffer("mask",
                             torch.triu(torch.ones(context_length,context_length),diagonal=1))
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim_out,embed_dim_out)

    def forward(self,x):
        batch,num_tokens,d_in = x.shape

        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)

        # reshaping the matrix for multi head attention computation
        query = query.view(batch,num_tokens,self.num_heads,self.head_dim)
        key = key.view(batch,num_tokens,self.num_heads,self.head_dim)
        value = value.view(batch,num_tokens,self.num_heads,self.head_dim)

        #Transposes from shape (b, num_tokens,num_heads, head_dim) to (b, num_heads,num_tokens, head_dim)
        query = query.transpose(1,2)
        key = key.transpose(1,2)
        value = value.transpose(1,2)

        # computing multi head attention scores

        attention_scores = query @ key.transpose(2,3)
        attention_scores.masked_fill_(self.mask.bool()[:num_tokens,:num_tokens],-torch.inf)
        attention_scores = attention_scores / x.shape[-1]**0.5

        attention_weights = attention_scores.softmax(dim=-1)
        attention_weights = self.dropout(attention_weights) # dropping some activation

        attention_vectors = (attention_weights @ value).transpose(1,2)

        attention_vectors = attention_vectors #Tensor shape:(b, num_tokens,n_heads,head_dim)
        attention_vectors = attention_vectors.contiguous().view(batch,num_tokens,self.embed_dim_out)

        attention_vectors = self.out_proj(attention_vectors)
        return attention_vectors