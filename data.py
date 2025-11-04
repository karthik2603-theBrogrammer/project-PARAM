from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

from utils import logger


FILE_PATH = "dataset.txt"
CONTEXT_SIZE = 256
STRIDE = 256
BATCH_SIZE = 4

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, context_size, stride):
        self.input_ids = []
        self.target_ids = []

        tokenized_text = tokenizer.encode(text)
        logger.info(f"Running Job with Total Tokens: {len(tokenized_text)}")
        for i in range(0, len(tokenized_text) - context_size, stride):
            input_chunk = tokenized_text[i: i + context_size]
            output_chunk = tokenized_text[i + 1: i + context_size + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(output_chunk))
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.target_ids[idx])

def get_dataloder(text, context_size, stride, train_ratio = 0.9, batch_size = 4, drop_last = True, shuffle = True, num_workers = 0):
    # tokenizer = tiktoken.get_encoding("gpt2") ## Will run with network access.
    with open('tiktoken_gpt2.pkl', 'rb') as f:  ## Use this when no-network access.
        tiktoken_gpt2 = pickle.load(f)
    tokenizer = tiktoken.core.Encoding(tiktoken_gpt2.pop('name'), **tiktoken_gpt2)
    split_point = int(len(text) * train_ratio)
    train_text = text[:split_point]
    val_text = text[split_point:]

    train_dataset = GPTDataset(train_text, tokenizer, context_size, stride)
    train_dataloader =  DataLoader(
        train_dataset, 
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
    )

    val_dataset = GPTDataset(val_text, tokenizer, context_size, stride)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
    )

    return train_dataloader, val_dataloader




if __name__ == "__main__":
    logger.info(f"Tiktoken version: {version('tiktoken')}")
    with open(FILE_PATH, "r", encoding = "utf-8") as f:
        raw_text = f.read()
    train_dataloader, _ = get_dataloder(text= raw_text, context_size= CONTEXT_SIZE, stride= STRIDE, batch_size= BATCH_SIZE)
    train_dataloader = iter(train_dataloader)
    logger.info(f"Dataset with Context Size: {CONTEXT_SIZE}, Stride: {STRIDE}, Batch Size: {BATCH_SIZE} and Total Batches/ Items: {len(train_dataloader)}")
    x, y = next(train_dataloader)
    logger.info(f"Train Sample: {x.shape}, {y.shape}")
