import torch
from model.GPT import GPTModel,generate_text
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from utils import compute_loss_batch, compute_loss_loader,text_to_token_ids,token_ids_to_text

def train_model(
        model:GPTModel,
        train_dataloader:DataLoader,
        val_dataloader:DataLoader,
        optimizer:Optimizer,
        device:str,
        num_epochs:int,
        eval_freq:int,
        eval_iter:int,
        start_context:str,
        tokenizer
):
    train_losses, val_losses,track_tokens_seen = [], [], []
    tokens_seen,global_step = 0, -1
    for epoch in range(num_epochs):
        for input_batch,target_batch in train_dataloader:
            loss = compute_loss_batch(input_batch,target_batch,model,device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step +=1
            tokens_seen += input_batch.numel()

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_dataloader, val_dataloader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )
        generate_and_print_sample(
            model, tokenizer, device, start_context
            )
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(
                    model:GPTModel,
                    train_dataloader:DataLoader,
                    val_dataloader:DataLoader, 
                    device:str, 
                    eval_iter:int):
    model.eval()
    with torch.no_grad():
        loss_train = compute_loss_loader(train_dataloader,model,device,eval_iter)
        loss_val = compute_loss_loader(val_dataloader,model,device,eval_iter)
        model.train()
    return loss_train,loss_val

def generate_and_print_sample(
            model:GPTModel, 
            tokenizer, 
            device:str, 
            start_context:str
            ):
    model.eval()
    context_size = model.positional_embedding.weight.shape[0]
    encoded = text_to_token_ids(start_context,tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(model,encoded,max_new_tokens=50,context_size=context_size)
    decoded_text = token_ids_to_text(token_ids,tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()