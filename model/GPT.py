import torch
import torch.nn as nn
from model.attention import MultiHeadAttention


class DummyGPTModel(nn.Module):
    def __init__(self, config:dict):
        super(DummyGPTModel,self).__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"],config["emb_dim"])
        self.positional_embedding = nn.Embedding(config["context_length"],config["emb_dim"])
        self.drop_embedding = nn.Dropout(config["drop_rate"])
        self.transformer_block = nn.Sequential([
            DummyTransformerBlock(config) for _ in range(config["n_layers"])
        ])

        self.final_normalization = DummyLayerNorm(config["emb_dim"])
        self.projection = nn.Linear(
            in_features=config["emb_dim"],
            out_features=config["vocab_size"]
        )

    def forward(self,in_idx:torch.Tensor):
        _,seq_len = in_idx.shape
        token_embedding = self.token_embedding(in_idx)
        positional_embedding = self.positional_embedding(
            torch.arange(seq_len,device=in_idx.device)
        )

        x = token_embedding + positional_embedding
        x = self.drop_embedding(x)
        x = self.transformer_block(x)
        x = self.final_normalization(x)
        x = self.projection(x)
        return x



class DummyTransformerBlock(nn.Module):
    def __init__(self, config):
        super(DummyTransformerBlock,self).__init__()

    def forward(self,x):
        return x
    

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
    
class LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super(LayerNorm,self).__init__()
        self.eps=1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self,x:torch.Tensor):
        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(dim=-1,keepdim=True,unbiased=False)
        x = (x-mean) /( std + self.eps )
        return x*self.scale + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
class FeedForward(nn.Module):
    def __init__(self, config:dict):
        super(FeedForward,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=config["emb_dim"],out_features=4 * config["emb_dim"]),
            GELU(),
            nn.Linear(in_features=4 * config["emb_dim"],out_features=config["emb_dim"])
        )

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, config:dict):
        super(TransformerBlock,self).__init__()
        self.attention = MultiHeadAttention(
            embde_dim_in=config["emb_dim"],
            embed_dim_out=config["emb_dim"],
            context_length=config["context_length"],
            num_heads=config["n_heads"],
            dropout=config["drop_rate"],
            bias=config["qkv_bias"]
        )
        self.feed_forward = FeedForward(config)
        self.layer_norm_1 = LayerNorm(config["emb_dim"])
        self.layer_norm_2 = LayerNorm(config["emb_dim"])
        self.dropout_shortut = nn.Dropout(config["drop_rate"])

    def forward(self,x:torch.Tensor):
        shortcut = x
        x = self.layer_norm_1(x)
        x = self.attention(x)
        x = self.dropout_shortut(x)
        x = x + shortcut

        shortcut = x 
        x = self.layer_norm_2(x)
        x = self.feed_forward(x)
        x = self.dropout_shortut(x)
        x = x + shortcut

        return x
    
class GPTModel(nn.Module):
    def __init__(self,config:dict):
        super(GPTModel,self).__init__()
        self.token_embedding = nn.Embedding(num_embeddings=config["vocab_size"],embedding_dim=config["emb_dim"])
        self.positional_embedding = nn.Embedding(num_embeddings=config["context_length"],embedding_dim=config["emb_dim"])
        self.drop_embedding = nn.Dropout(config["drop_rate"])
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])]
        )
        self.final_norm = LayerNorm(config['emb_dim'])
        self.projection = nn.Linear(in_features=config['emb_dim'],out_features=config["vocab_size"],bias=False)

    def forward(self,in_idx:torch.Tensor):
        _,seq_len = in_idx.shape
        token_embeddings = self.token_embedding(in_idx)
        positional_embedding = self.positional_embedding(torch.arange(seq_len,device=in_idx.device))
        x = token_embeddings + positional_embedding

        x = self.drop_embedding(x)

        x = self.transformer_blocks(x)

        x = self.final_norm(x)

        logits = self.projection(x)

        return logits

def generate_text(
        model:GPTModel,
        idx:torch.Tensor,
        max_new_tokens:int,
        context_size:int
):
    for _ in range(max_new_tokens):
        current_idx = idx[:,-context_size:] # takes only at max the context size
        with torch.no_grad():
            logits:torch.Tensor = model(current_idx)
        logits = logits[:,-1,:] # take the last row which will tell us the predicted token, of shape (batch,vocab_size)
        proba = logits.softmax(dim=-1) # shape (batch,vocab_size)
        next_idx = proba.argmax(dim=-1,keepdim=True) # shape (batch,1)

        idx = torch.cat([idx,next_idx],dim=-1) # Appends sampled index to the running sequence
    return idx

def generate(model:GPTModel,
        idx:torch.Tensor,
        max_new_tokens:int,
        context_size:int,
        temperature=0.0,
        top_k:int=None,
        eos_id:str=None
        ):
    for _ in range(max_new_tokens):
        input_ids = idx[:-context_size]
        with torch.no_grad():
            logits:torch.Tensor = model(input_ids)
        if top_k is not None:
            top_logits,_ = torch.topk(logits,top_k)
            min_val = top_logits[-1]
            logits = logits.where(
                logits >= min_val,
                -torch.inf
            ).to(logits.device)
        if temperature > 0.0:
            logits = logits / temperature
            probas = logits.softmax(dim=-1)
            next_index = torch.multinomial(probas,num_samples=1)
        else:
            next_index = logits.argmax(dim=-1,keepdim=True)
        
        if next_index== eos_id:
            break

        idx = torch.cat([idx,next_index],dim=-1)
    return idx