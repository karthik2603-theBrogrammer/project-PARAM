from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import pickle

from utils import logger, format_large_number


FILE_PATH = "dataset.txt"
CONTEXT_SIZE = 256
STRIDE = 256
BATCH_SIZE = 4

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, context_size, stride):
        self.input_ids = []
        self.target_ids = []

        tokenized_text = tokenizer.encode(text)
        logger.info(f"Running Job with Total Tokens: {format_large_number(len(tokenized_text))}; Sample: {tokenized_text[:10]}....")
        for i in range(0, len(tokenized_text) - context_size, stride):
            input_chunk = tokenized_text[i: i + context_size]
            output_chunk = tokenized_text[i + 1: i + context_size + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(output_chunk))
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.target_ids[idx])

def get_ddp_dataloader(text, context_size, stride, batch_size, world_size, rank):

    split_point = int(len(text) * 0.9)
    train_text = text[:split_point]
    val_text = text[split_point:]
    with open('tiktoken_gpt2.pkl', 'rb') as f:
        tiktoken_gpt2 = pickle.load(f)
    tokenizer = tiktoken.core.Encoding(tiktoken_gpt2.pop('name'), **tiktoken_gpt2)

    train_dataset = GPTDataset(train_text, tokenizer, context_size, stride)
    val_dataset = GPTDataset(val_text, tokenizer, context_size, stride)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, # Per GPU batch size in DDP.
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    with open("dataset.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()
    logger.info(f"Dataset length: {len(raw_text)} characters.")
    logger.info("[DDP Data] Ready to use.")