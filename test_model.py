import torch
import pickle
import tiktoken
from model import ParamModel, PARAM_CONFIG_1B
from utils import generate_and_print_sample
model_file = "/scratch/karthick/pretrain/param-7b/param-checkpoints/20250220_173103_096d4b770e/checkpoint_step_480000.pt"

model = ParamModel(cfg = PARAM_CONFIG_1B)
model.load_state_dict(torch.load(model_file)['model_state_dict'])
model.to("cuda")

with open('tiktoken_gpt2.pkl', 'rb') as f:  ## Use this when no-network access.
    tiktoken_gpt2 = pickle.load(f)
    tokenizer = tiktoken.core.Encoding(tiktoken_gpt2.pop('name'), **tiktoken_gpt2)

generate_and_print_sample(
    model= model,
    tokenizer= tokenizer,
    device= "cuda",
    start_context= "hello, I am a"
)