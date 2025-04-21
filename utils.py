import tiktoken
from model.GPT import generate_text,GPTModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def text_to_token_ids(text:str,tokenizer):
    encoded = tokenizer.encode(text,allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids:torch.Tensor,tokenizer):
    token_ids = token_ids.squeeze(0).tolist()
    decoded = tokenizer.decode(token_ids)
    return decoded

def compute_loss_batch(input_bach:torch.Tensor,target_batch:torch.Tensor,model,device:str):
    input_bach = input_bach.to(device)
    target_batch = target_batch.to(device)
    predictions:torch.Tensor = model(input_bach)
    loss = F.cross_entropy(
        predictions.flatten(0,1),
        target_batch.flatten()
    )
    return loss

def compute_loss_loader(
        data_loader:DataLoader,
        model:GPTModel,
        device:str,
        num_batches:int=None
):
    total_losses = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches == None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches,len(data_loader))
    for i,(input_batch,target_batch) in enumerate(data_loader):
        if i<num_batches:
            loss = compute_loss_batch(input_batch,target_batch,model,device)
            total_losses+=loss.item()
        else:
            break
    return loss