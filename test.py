from model_hf import ParamModel, PARAM_CONFIG_MOE_477M
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "/scratch/karthick/pretrain/param-7b/llama3.1-tokenizer/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
)
if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
PARAM_CONFIG_MOE_477M["vocab_size"] = 128002

print(tokenizer.vocab_size)
model = ParamModel(PARAM_CONFIG_MOE_477M)
print(model)
print(model.get_parameters())
import torch
ip = torch.randint(0, PARAM_CONFIG_MOE_477M["vocab_size"], (1, 1024)) 
am = torch.ones_like(ip)
o = model(ip, am, ip)

print(o['loss'])


model = model.to("cuda")
model.generate(
    tokenizer= tokenizer,
    max_length = 1024,
    prompt= "Hello, ",
    num_tokens = 50,
    device= "cuda"
)