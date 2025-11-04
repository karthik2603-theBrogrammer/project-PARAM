# train.py (Renamed to distributed_trainer.py)
from model import ParamModel
import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
import argparse
import tiktoken
import pickle
import datetime
from num2words import num2words

from utils import text_to_token_ids, token_ids_to_text, calc_loss_loader, logger, evaluate_model, generate_and_print_sample, format_large_number, calculate_loss_batch
from data import get_dataloder


num_epochs = 200
CONTEXT_SIZE = 1024
STRIDE = 1024
BATCH_SIZE = 2
CHECKPOINT_INTERVAL = 2000
PARAM_CONFIG = {
    "vocab_size": 50257,
    "context_length": CONTEXT_SIZE,
    "emb_dim": 1536,
    "num_heads": 24,
    "num_layers": 24,
    "dropout_rate": 0.25,
    "qkv_bias":  False
}


# Parse arguments to initialize deepspeed
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


with open('tiktoken_gpt2.pkl', 'rb') as f:  ## Use this when no-network access.
    tiktoken_gpt2 = pickle.load(f)
    # BPE Encoder used for GPT based models.
    tokenizer = tiktoken.core.Encoding(tiktoken_gpt2.pop('name'), **tiktoken_gpt2)

model = ParamModel(cfg=PARAM_CONFIG)
criterion = nn.CrossEntropyLoss()


with open("dataset.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print(len(raw_text))

train_dataloader, val_dataloader = get_dataloder(text=raw_text, context_size=CONTEXT_SIZE, stride=STRIDE, batch_size=BATCH_SIZE)
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=model.parameters()
)
device = model_engine.device  
logger.info("Setup Deepspeed Model Engine.")
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Total Parameters in our model: {format_large_number(total_params)} Parameters.")

eval_freq = 100  
eval_iter = 200    
start_context = "Once upon a time"  
train_losses, val_losses, track_tokens_seen = [], [], []
tokens_seen, global_step = 0, 0
len_trainloader = len(train_dataloader)
total_steps = num_epochs * len_trainloader

for epoch in range(num_epochs):
    model_engine.train()
    
    for input_batch, target_batch in train_dataloader:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        
        loss = calculate_loss_batch(input_batch, target_batch, model_engine, device)
        model_engine.backward(loss)
        model_engine.step()
        
        tokens_seen += input_batch.numel()
        global_step += 1
        
        # Only print from main process
        if model_engine.local_rank == 0:
            print(f"\r Device: [{model_engine.local_rank}] Step: {global_step}/ {total_steps} ==== ", end="", flush=True)
        
        # Save model in regular intervals --> Checkpoint every 2000 steps.
        if global_step % CHECKPOINT_INTERVAL == 0 and model_engine.local_rank == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            model_engine.save_checkpoint("checkpoints", f"checkpoint-{global_step}-{timestamp}")
        
        if global_step % eval_freq == 0 and model_engine.local_rank == 0:
            train_loss, val_loss = evaluate_model(
                model_engine, train_dataloader, val_dataloader, device, eval_iter
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            track_tokens_seen.append(tokens_seen)
            logger.info(f"\nEp {epoch+1}, (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
            
            generate_and_print_sample(model_engine, tokenizer, device, "Who let the dogs out")
            generate_and_print_sample(model_engine, tokenizer, device, "I played football and")