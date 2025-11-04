import torch
import torch.nn as nn
import torch.nn.init as init
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Param 7B")


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text(
        model: nn.Module,
        inp: list,
        max_new_tokens: int,
        context_size: int,
):
    for _ in range(max_new_tokens):
        inp_truncate = inp[:,-context_size:]
        with torch.no_grad():
            logits = model(inp_truncate)
        logits = logits[:,-1,:]
        probs = torch.softmax(logits, dim = -1)
        new_token_id = torch.argmax(probs, dim= -1, keepdim= True)
        print(new_token_id)
        inp = torch.cat((inp, new_token_id), dim = 1)
    return inp


def calculate_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)

    loss = nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):

    if len(data_loader) == 0:
        logger.warning("Dataloader of size = 0 is given.")
        raise RuntimeError("Invalid Dataloder.")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    total_loss = 0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calculate_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


def train_model(model: nn.Module, 
                train_dataloader, 
                val_dataloader, 
                epochs, 
                optimizer, 
                device, 
                eval_freq, 
                eval_iter, 
                start_context, 
                tokenizer
            ):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, 0
    len_trainloader = len(train_dataloader) # Also gives the number of steps/ batches in an epoch.
    total_steps = epochs * len_trainloader
    for epoch in range(epochs):
        model.train()
        for input_batch, target_batch in train_dataloader:
            optimizer.zero_grad()
            loss = calculate_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            print(f"\r Step: {global_step}/ {total_steps} ==== ", end = "", flush= True)
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_dataloader, val_dataloader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                logger.info(f"\nEp {epoch+1}, (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                
                generate_and_print_sample(
                    model, tokenizer, device, start_context
                )
                generate_and_print_sample(
                    model, tokenizer, device, "I played football and "        
                )
    return train_losses, val_losses, track_tokens_seen


def evaluate_model(
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        device,
        eval_iter
):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_dataloader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_dataloader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(
    model: nn.Module, 
    tokenizer, 
    device, 
    start_context
):
    model.eval()
    context_size = model.position_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(
            model=model, inp= encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    print(tokenizer.decode([198, 447]))
    model.train()


def format_large_number(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.0f}B"  # Convert to billions
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.0f}M"  # Convert to millions
    elif num >= 1_000:
        return f"{num / 1_000:.0f}K"  # Convert to thousands
    else:
        return str(num)


def init_weights(module):
    if isinstance(module, nn.Linear):
        # Xavier (Glorot) uniform initialization for weights
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)