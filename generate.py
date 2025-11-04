import torch
from safetensors.torch import load_file
state_dict = load_file("/scratch/karthick/pretrain/param-7b/autoregressive_model_output/checkpoint-20000/model.safetensors", device="cpu")
from model_hf import ParamModel, PARAM_CONFIG_MOE_477M
PARAM_CONFIG_MOE_477M['vocab_size'] = 128002
model = ParamModel(
    cfg= PARAM_CONFIG_MOE_477M
 )
model.load_state_dict(state_dict)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/scratch/karthick/pretrain/param-7b/llama3.1-tokenizer/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b")
prompt = "hi,"
model = model.to("cuda")
model.eval()
encodings = tokenizer(prompt, return_tensors = 'pt', add_special_tokens = True)
input_ids = encodings["input_ids"].to("cuda")
print(prompt, end="")
for _ in range(200):
         with torch.no_grad():
             outputs = model(
                 input_ids = input_ids,
                 # attention_mask = torch.ones_like(input_ids)
             )
             logits = outputs["logits"][0, -1, : ]
             next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
             print(tokenizer.decode([next_token.item()]), end='')
             input_ids = torch.cat([input_ids, next_token], dim=1)
             input_ids = input_ids[:, -PARAM_CONFIG_MOE_477M['context_length']:]
             if next_token.item() == tokenizer.eos_token_id:
                 break