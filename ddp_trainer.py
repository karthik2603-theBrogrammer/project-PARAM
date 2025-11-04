import os
import uuid
import time
import shutil
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import logging
import pickle
import tiktoken
from datetime import datetime, timedelta
from model import ParamModel, PARAM_CONFIG_1B, PARAM_CONFIG_124M
from utils import (
    text_to_token_ids, token_ids_to_text, calc_loss_loader,
    logger, train_model, format_large_number, calculate_loss_batch, generate_and_print_sample
)
from ddp_data import get_ddp_dataloader

def setup_distributed():
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return world_size, rank, local_rank

def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load checkpoint and return training state"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Return training state
    return (
        checkpoint['epoch'],
        checkpoint['global_step'],
        checkpoint['tokens_seen'],
        checkpoint['train_losses'],
        checkpoint['val_losses']
    )

def train_ddp_model(
    model, train_loader, val_loader, epochs, optimizer,
    device, eval_freq, checkpoint_freq, eval_iter, tokenizer,
    rank, resume_checkpoint=None, start_epoch=0, start_step=0,
    initial_tokens_seen=0, initial_train_losses=None, initial_val_losses=None
):
    train_losses = initial_train_losses or []
    val_losses = initial_val_losses or []
    track_tokens_seen = []
    tokens_seen = initial_tokens_seen
    global_step = start_step
    len_trainloader = len(train_loader)
    total_steps = epochs * len_trainloader

    checkpoint_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_suffix = uuid.uuid4().hex[:10]
    CHECKPOINT_UUID = f"{checkpoint_prefix}_{checkpoint_suffix}"

    if resume_checkpoint:
        # If resuming, use the checkpoint's directory
        CHECKPOINT_UUID = os.path.basename(os.path.dirname(resume_checkpoint))

    logger.info(f"[DDP Rank: {rank}]; Train Loader Size: {len(train_loader)}")
    start_time = time.perf_counter()
    
    for epoch in range(start_epoch, epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        for input_batch, target_batch in train_loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad()
            loss = calculate_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss / dist.get_world_size()
            tokens_seen += input_batch.numel() * dist.get_world_size()
            global_step += 1

            if rank == 0:  
                print(f"\r Step: {global_step}/ {total_steps} ==== ", end="", flush=True)

            if global_step % checkpoint_freq == 0 and rank == 0:
                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "tokens_seen": tokens_seen,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                }

                checkpoint_dir = f"param-checkpoints/{CHECKPOINT_UUID}"
                os.makedirs(checkpoint_dir, exist_ok=True)

                checkpoint_file = f"checkpoint_step_{global_step}.pt"
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"\nCheckpoint saved at step {global_step} to {checkpoint_path}")

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_ddp_model(model, train_loader, val_loader, device, eval_iter)
                if rank == 0: 
                    end_time = time.perf_counter()
                    time_taken_for_steps = end_time - start_time
                    time_per_step = time_taken_for_steps / eval_freq
                    time_remaining = (total_steps - global_step) * time_per_step
                    sec = timedelta(seconds=time_remaining)
                    d = datetime(1,1,1) + sec
                    start_time = time.perf_counter()

                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    logger.info(f"\nEp {epoch+1}, (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f};")
                    logger.info(f"Eval freq time: {time_taken_for_steps}; ETA: {d.day - 1} Days, {d.hour} Hours, {d.minute} Minutes, {d.second} Seconds.")
                    generate_and_print_sample(model.module, tokenizer, device, "He came into the room and")
                    generate_and_print_sample(model.module, tokenizer, device, "I played football and ")

    return train_losses, val_losses, track_tokens_seen

def evaluate_ddp_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

if __name__ == "__main__":
    # Add checkpoint path argument
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_checkpoint', type=str, help='Path to checkpoint file to resume from')
    args = parser.parse_args()

    world_size, rank, local_rank = setup_distributed()
    logger.info(f"[DDP; rank {local_rank}] Running DDP with World Size: {world_size}")
    device = f"cuda:{local_rank}"
    
    CONTEXT_SIZE = 512
    STRIDE = 512
    BATCH_SIZE = 2
    num_epochs = 2500

    if rank == 0:
        with open("dataset_small.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()
        logger.info(f"Dataset length: {len(raw_text)} characters.")
    
    if rank == 0:
        text_length = torch.tensor(len(raw_text), dtype=torch.long, device=device)
    else:
        text_length = torch.tensor(0, dtype=torch.long, device=device)
    dist.broadcast(text_length, src=0)
    
    if rank == 0:
        text_tensor = torch.tensor([ord(c) for c in raw_text], dtype=torch.long, device=device)
    else:
        text_tensor = torch.empty(text_length.item(), dtype=torch.long, device=device)
    dist.broadcast(text_tensor, src=0)
    
    if rank != 0:
        raw_text = ''.join([chr(i) for i in text_tensor.cpu().tolist()])
        
    train_loader, val_loader, tokenizer = get_ddp_dataloader(
        raw_text, CONTEXT_SIZE, STRIDE, BATCH_SIZE, world_size, rank
    )

    model = ParamModel(cfg=PARAM_CONFIG_124M).to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
        weight_decay=0.1
    )

    # Initialize training state variables
    start_epoch = 0
    start_step = 0
    initial_tokens_seen = 0
    initial_train_losses = []
    initial_val_losses = []

    # Load checkpoint if provided
    if args.resume_checkpoint:
        if rank == 0:
            logger.info(f"Resuming training from checkpoint: {args.resume_checkpoint}")
        start_epoch, start_step, initial_tokens_seen, initial_train_losses, initial_val_losses = load_checkpoint(
            args.resume_checkpoint, model, optimizer, device
        )

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total Parameters: {format_large_number(total_params)}")
        logger.info(f"[DDP] Beginning distributed training from step {start_step}...")

    train_losses, val_losses, tokens_seen = train_ddp_model(
        model, train_loader, val_loader, num_epochs, optimizer, device,
        eval_freq=100, eval_iter=100, checkpoint_freq=20000, tokenizer=tokenizer, rank=rank,
        resume_checkpoint=args.resume_checkpoint, start_epoch=start_epoch, start_step=start_step,
        initial_tokens_seen=initial_tokens_seen, initial_train_losses=initial_train_losses,
        initial_val_losses=initial_val_losses
    )
    
    dist.destroy_process_group()