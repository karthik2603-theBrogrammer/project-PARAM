import torch
from model import ParamModel
import tiktoken
import pickle
from utils import text_to_token_ids, token_ids_to_text, calc_loss_loader, logger, train_model, format_large_number
from data import get_dataloder
from num2words import num2words

device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 6500
CONTEXT_SIZE = 256
STRIDE = 256
BATCH_SIZE = 2
PARAM_CONFIG = {
    "vocab_size": 50257,
    "context_length": CONTEXT_SIZE,
    "emb_dim": 1536,
    "num_heads": 24,
    "num_layers": 24,
    "dropout_rate": 0.25,
    "qkv_bias":  False
}

with open("dataset.txt", "r", encoding = "utf-8") as f:
    raw_text = f.read()
print(len(raw_text))

train_dataloader, val_dataloader = get_dataloder(text= raw_text, context_size= CONTEXT_SIZE, stride= STRIDE, batch_size= BATCH_SIZE)

with open('tiktoken_gpt2.pkl', 'rb') as f:  ## Use this when no-network access.
    tiktoken_gpt2 = pickle.load(f)
    tokenizer = tiktoken.core.Encoding(tiktoken_gpt2.pop('name'), **tiktoken_gpt2)

model = ParamModel(cfg= PARAM_CONFIG)
model.to(device= device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.01, weight_decay=0.1
)
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Total Parameters in our model: {format_large_number(total_params)} Parameters.")
logger.info(f"Beginning pre-training of Param Model...")
train_losses, val_losses, tokens_seen = train_model(
    model, train_dataloader, val_dataloader, num_epochs, optimizer, device,
    eval_freq=5000, eval_iter= 100,
    start_context="He came into the room and", tokenizer=tokenizer
)
