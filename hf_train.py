#  29825087

# 24738918 number of examples in _train file
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, TrainerCallback
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset, load_from_disk
from model_hf import ParamModel, PARAM_CONFIG_MOE_477M
from utils import logger, init_weights
from safetensors.torch import load_file


CONTINUAL_PRETRAINING = True
config = PARAM_CONFIG_MOE_477M
CONTEXT_SIZE = config["context_length"]

class CustomEvaluationCallback(TrainerCallback):
    def __init__(self, eval_function, eval_steps=1000):
        self.eval_function = eval_function
        self.eval_steps = eval_steps
        self.best_metric = float('-inf')
        
    def on_log(self, args, state, control, **kwargs):
        # Check if it's time to run evaluation
        if state.global_step % self.eval_steps == 0:
            model = kwargs.get('model')
            tokenizer = kwargs.get('tokenizer', None)
            
            # Run your custom evaluation
            eval_results = self.eval_function(model, tokenizer)
            
class ParamAutoregressiveDataset(Dataset):
    def __init__(self, dataset, tokenizer, context_size = 1024):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.context_size = context_size

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        encodings = self.tokenizer (
            text,
            truncation = True,
            max_length = self.context_size,
            padding = "max_length",
            return_tensors = "pt",
            add_special_tokens=True
        )
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }
class ParamStreamingDataset(IterableDataset):
    def __init__(self, data_file_path, tokenizer, total_items: int, context_size = CONTEXT_SIZE, shuffle_buffer_size = 10000):
        self.data_file_path = data_file_path
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.total_items = total_items
    
    def clean_text(self, text):
        import re
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'Edited:.*', '', text)
        text = re.sub(r'http\S+', '', text)

        return text.strip()
    
    def __iter__(self):
        buffer = []
        with open(self.data_file_path, 'r', encoding= 'utf-8') as file:
            for line in file:
                if line.strip():
                    try:
                        data = json.loads(line)
                        text = data.get("text", "")

                        if text:
                            if self.shuffle_buffer_size <= 1:
                                processed_text = self._process_text(text)
                                yield processed_text
                            else:
                                buffer.append(text)
                                if len(buffer) >= self.shuffle_buffer_size:
                                    indices = torch.randperm(len(buffer))
                                    for idx in indices:
                                        yield self._process_text(buffer[idx])
                                    buffer = []

                    except json.JSONDecodeError:
                        continue

        if buffer and self.shuffle_buffer_size > 1:
            indices = torch.randperm(len(buffer))
            for idx in indices:
                yield self._process_text(buffer[idx])

    def __len__(self):
        return self.total_items

    def _process_text(self, text):
        tokenizer_encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.context_size,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True
        )
        input_ids = tokenizer_encodings["input_ids"].squeeze()
        attention_mask = tokenizer_encodings["attention_mask"].squeeze()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }
                    
class CustomTrainer(Trainer):
    def _save(self, output_dir=None, state_dict=None):
        from safetensors.torch import save_file
        
        # Get state dict
        if state_dict is None:
            state_dict = self.model.state_dict()
            
        # Create a deep copy of tensors that share memory
        for name, param in list(state_dict.items()):
            if name in ['out_head.weight', 'token_emb.weight']:
                state_dict[name] = param.clone()
                
        # Save using safetensors
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
        
        # Save tokenizer and model config
        if hasattr(self.model, "config"):
            self.model.config.save_pretrained(output_dir)
    
def evaluate_during_training(model, tokenizer, prompt, max_length = 50):

    model.eval()
    encodings = tokenizer(
        prompt, 
        return_tensors = "pt", 
        add_special_tokens=True
    )
    input_ids = encodings["input_ids"].to("cuda")
    attention_mask = encodings["attention_mask"].to("cuda")


    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(
                input_ids = input_ids,
                # attention_mask = torch.ones_like(input_ids)
            )
            logits = outputs["logits"][0, -1, : ]
            next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
            print(tokenizer.decode([next_token]))
            input_ids = torch.cat([input_ids, next_token], dim=1)
            input_ids = input_ids[:, -CONTEXT_SIZE:]
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(
        input_ids[0]
    )


def train(device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(
        "/scratch/karthick/pretrain/param-7b/llama3.1-tokenizer/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
    )

    # Some tokenizer's dont have the pad_token set by default, hence set as eos_token. 
    config["vocab_size"] = tokenizer.vocab_size 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config["vocab_size"] = 128002
    logger.info("Tokenizer ready.")


    streaming_train_ds = ParamStreamingDataset(
        data_file_path= "/scratch/karthick/pretrain/param-7b/paramds_train_1.jsonl",
        tokenizer= tokenizer,
        total_items= 23000000,
        context_size= CONTEXT_SIZE,
        shuffle_buffer_size= 1
    )
    val_ds = load_dataset(
        'json', 
        data_files = [
            '/scratch/karthick/pretrain/param-7b/paramds_val.jsonl'
        ]
    )['train']

    val_ds = ParamAutoregressiveDataset(
        dataset= val_ds,
        tokenizer= tokenizer,
        context_size= CONTEXT_SIZE
    )
    # streaming_val_ds = ParamStreamingDataset(
    #     data_file_path= "",
    #     tokenizer= tokenizer,
    #     total_items= 100,
    #     context_size= CONTEXT_SIZE,
    #     shuffle_buffer_size= 1
    # )
    logger.info("Dataset Ready.")
    model = ParamModel(
        cfg= config
    )
    logger.info(f"Launching param training with total model parmeters: {model.get_parameters()}")

    # if CONTINUAL_PRETRAINING:
    #     state_dict = load_file("/scratch/karthick/pretrain/param-7b/autoregressive_model_output/checkpoint-40000/model.safetensors", device="cpu")
    #     model.load_state_dict(state_dict) 
    # else:
    #     # Training the model from scratch - Apply glorot/ xavier initialization
    #     # where the linear layers are initialized such that the model converges
    #     # faster.
    #     # NOTE: Weight initialization being done inherently 
    #     # based on ST-MoE and Switch transformer.
    #     pass

    #     # legacy: model.apply(init_weights)
    logger.info(model)
    model = model.to(device)


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  
    )

    training_args = TrainingArguments(
        output_dir="./autoregressive_model_output",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=8,
        save_steps=10000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
        logging_first_step = True,
        evaluation_strategy="steps",  # When to evaluate (no, steps, epoch)
        eval_steps=300, 
        prediction_loss_only=True,
        report_to=["tensorboard"],

        dataloader_num_workers = 1,
        dataloader_pin_memory = True,

        # Mixed precision settings,
        fp16 = True,
        fp16_backend = "amp",
        

        dispatch_batches=None, 
        dataloader_prefetch_factor=None  
    )

    def evaluate_model(model, _):
        result = model.generate(
            tokenizer = tokenizer, 
            max_length = 1024,
            prompt= "Hello, ",
            num_tokens = 50,
            device= "cuda",
            stream = True
        )
        # print(f"\nGenerated text at step {trainer.state.global_step}:\n{result}\n")
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset = streaming_train_ds,
        eval_dataset = val_ds,
        data_collator = data_collator,
        callbacks=[
            CustomEvaluationCallback(
                eval_function=evaluate_model,
                eval_steps=300
            )
        ]
    )
    if CONTINUAL_PRETRAINING:
        trainer.train(
            resume_from_checkpoint="./autoregressive_model_output/checkpoint-230000"
        )
    else:
        trainer.train()
    trainer.save_model("./final_param_model_734M")
    return model, tokenizer


if __name__ == "__main__":
    train(device= "cuda")
