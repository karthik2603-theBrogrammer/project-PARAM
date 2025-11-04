import math
import torch
import torch.nn as nn
from transformer_block import TransformerBlock
from layer_norm import LayerNorm
from moe import MLPExperts
from typing import Optional
from manager import MANAGER


class ParamModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        assert cfg["vocab_size"] is not None
        assert cfg["context_length"] is not None
        self.cfg = cfg

        
        
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.position_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout_emb = nn.Dropout(cfg['dropout_rate'])

        if cfg["num_experts"] == 1:
            # create normal transformer blocks
            blocks = nn.ModuleList([TransformerBlock( cfg= cfg) for _ in range(cfg["num_layers"])])
        else:
            blocks = []
            for i in range(cfg["num_layers"]):
                use_moe = (i % cfg["stride"]) == 0
                blocks.append(TransformerBlock(cfg, use_moe=use_moe))
            blocks = nn.ModuleList(blocks)

        self.transformer_blocks = nn.ModuleList(blocks)
        self.final_norm = LayerNorm(emb_dim= cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias= False)

        self.token_emb.weight = self.out_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        # optionall use switch transformer special init scheme for experts
        # See pg. 10 here: https://arxiv.org/abs/2101.03961
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('experts.c_proj'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * cfg["num_layers"]))
    
    @torch.no_grad()
    def _init_weights(self, module):
        # optionally use switch transformer-style initialization
        # see page 10 for switch init explanation: https://arxiv.org/abs/2101.03961
        if isinstance(module, nn.Linear):
            if self.cfg["use_switch_tfm_init"]:
                scale = self.cfg["switch_tfm_init_scale"]

                # linear layers have flipped dimensions in torch
                # size of weights is [out_dim, in_dim] 
                w_fan_in = module.weight.shape[-1]
                w_std = (scale / w_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=w_std,
                    a=-2*w_std,
                    b=2*w_std,
                )
            else:
                # perform standard (normal) initialization of weights
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            # always initialize bias to zero
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, MLPExperts):
            # we have to init expert weights manually because
            # nn.Parameter is not a type of module in torch
            if self.cfg["use_switch_tfm_init"]:
                scale = self.cfg["switch_tfm_init_scale"]

                c_fc_fan_in = module.c_fc.shape[-2]
                c_fc_std = (scale / c_fc_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.c_fc,
                    mean=0.0,
                    std=c_fc_std,
                    a=-2*c_fc_std,
                    b=2*c_fc_std,
                )

                c_proj_fan_in = module.c_proj.shape[-2]
                c_proj_std = (scale / c_proj_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.c_proj,
                    mean=0.0,
                    std=c_proj_std,
                    a=-2*c_proj_std,
                    b=2*c_proj_std,
                )
            else:
                # perform standard (normal) initialization of weights
                torch.nn.init.normal_(module.c_fc, mean=0.0, std=0.02)
                torch.nn.init.normal_(module.c_proj, mean=0.0, std=0.02)

            # bias is always initialized to zero
            if module.fc_bias is not None:
                torch.nn.init.zeros_(module.fc_bias)
                torch.nn.init.zeros_(module.proj_bias)
        elif isinstance(module, nn.Embedding):
            # just use standard initialization scheme for embedding always
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    
    def get_parameters(self):
        params = sum(p.numel() for p in self.parameters())
        if params >= 1000000000:
            return f"{round(params/ 1000000000, 2)}B"
        elif params >= 1000000:
            return f"{round(params/ 1000000, 2)}M"
        else:
            return f"{round(params/1000, 2)}K"

    def forward(self, input_ids, attention_mask: Optional[torch.tensor] = None, labels = None):
        b, num_tokens = input_ids.shape
        token_emb = self.token_emb(input_ids)
        position_emb = self.position_emb(torch.arange(num_tokens, device= input_ids.device))
        position_emb = position_emb.unsqueeze(0).expand(b, -1, -1)  # Expand to [b, num_tokens, emb_dim]

        x = token_emb + position_emb
        x = self.dropout_emb(x)
        for module in self.transformer_blocks:
            x = module(x, attention_mask = attention_mask)
        x = self.final_norm(x)
        logits = self.out_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) # Standard neural network loss

            if self.cfg["num_experts"] > 1 and self.cfg["use_aux_loss"]:
                loss += self.cfg["aux_loss_weight"] * MANAGER.aggregate_aux_loss()
                MANAGER.reset_aux_loss()
            if self.cfg["num_experts"] > 1 and self.cfg["use_aux_loss"]:
                loss += self.cfg["router_z_loss_weight"] * MANAGER.aggregate_router_z_loss()
                MANAGER.reset_router_z_loss()

        return {
            "loss": loss,
            "logits": logits,
        } if loss is not None else {
            "logits": logits
        }

    @torch.no_grad
    def generate(
        self,
        tokenizer,
        prompt,
        max_length,
        num_tokens = 50,
        temperature = 1,
        top_k = None,
        stream = True,
        device = "cpu"
    ):
        encodings = tokenizer(
            prompt, 
            max_length = max_length,
            padding = "max_length",
            return_tensors = "pt", 
            add_special_tokens=True
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        for _ in range(num_tokens):
            outputs = self(
                input_ids = input_ids,
                attention_mask = attention_mask
            )
            logits = outputs["logits"][0, -1, : ]
            next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
            if stream:
                print(tokenizer.decode([next_token.item()]), end= "")
            input_ids = torch.cat([input_ids, next_token], dim=1)
            input_ids = input_ids[:, -self.cfg["context_length"]:]
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        return tokenizer.decode(
            input_ids[0]
        )
        
class ParamModelForCausalLM(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.position_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout_emb = nn.Dropout(cfg['dropout_rate'])

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(cfg= cfg) for _ in range(cfg["num_layers"])
        ])
        self.final_norm = LayerNorm(emb_dim= cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias= False)
    
    def get_parameters(self):
        params = sum(p.numel() for p in self.parameters())
        if params >= 1000000000:
            return f"{round(params/ 1000000000, 2)}B"
        elif params >= 1000000:
            return f"{round(params/ 1000000, 2)}M"
        else:
            return f"{round(params/1000, 2)}K"

    def forward(self, input_ids, attention_mask: Optional[torch.tensor] = None, labels = None):
        b, num_tokens = input_ids.shape
        token_emb = self.token_emb(input_ids)
        position_emb = self.position_emb(torch.arange(num_tokens, device= input_ids.device))
        position_emb = position_emb.unsqueeze(0).expand(b, -1, -1)  # Expand to [b, num_tokens, emb_dim]

        x = token_emb + position_emb
        x = self.dropout_emb(x)
        for module in self.transformer_blocks:
            x = module(x, attention_mask = attention_mask)
        x = self.final_norm(x)
        logits = self.out_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) # Standard neural network loss

        return {
            "loss": loss,
            "logits": logits,
        } if loss is not None else {
            "logits": logits
        }


PARAM_CONFIG_124M = {
    "vocab_size": 128001, # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768, # Embedding dimension
    "num_heads": 12, # Number of attention heads
    "num_layers": 12, # Number of layers
    "dropout_rate": 0.1, # Dropout rate
    "qkv_bias": False, # Query-Key-Value bias
    "top_k": 3,
    "num_experts": 8,
    "aux_loss_coefficient": 0.005
}

PARAM_CONFIG_734M = {
    "vocab_size": 128000, 
    "context_length": 1024, 
    "emb_dim": 1536,
    "num_heads": 24, 
    "num_layers": 12, 
    "dropout_rate": 0.1, 
    "qkv_bias": False,
    "top_k": 3,
    "num_experts": 8,
    "aux_loss_coefficient": 0.005
}

PARAM_CONFIG_1_5B = {
    "vocab_size": 128001,
    "context_length": 1024,
    "emb_dim": 1536,
    "num_heads": 24,
    "num_layers": 36,
    "dropout_rate": 0.25,
    "qkv_bias": False,
    "top_k": 3,
    "num_experts": 8,
    "aux_loss_coefficient": 0.005
}

PARAM_CONFIG_MOE_2_4B= {
    "vocab_size": 128000, 
    "context_length": 1024, 
    "emb_dim": 1536,
    "num_heads": 24, 
    "num_layers": 12, 
    "dropout_rate": 0.1, 
    "qkv_bias": False,
    "top_k": 3,
    "num_experts": 8,
    "aux_loss_coefficient": 0.005
}

PARAM_CONFIG_MOE_477M = {
    "vocab_size": 128000, 
    "context_length": 1024, 
    "emb_dim": 768, 
    "num_heads": 12, 
    "num_layers": 16, 
    "dropout_rate": 0.1, 
    "qkv_bias": False, 
    "bias": True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    
    # MoE-related configs 
    "top_k": 3,
    "num_experts": 8,
    "use_aux_loss": True, # apply auxiliary loss (from Switch Transformer) in router
    "use_router_z_loss" : True, # apply router z loss (from ST-MoE)
    "use_noisy_top_k" : True,
    "aux_loss_weight":  0.01, # default setting from Switch Transformer (see top of page 8)
    "router_z_loss_weight":  0.001, # default setting from ST-MoE (see page 8 eq. 6)
    "train_capacity" : 1.25,  # default setting from ST-MoE (see top of page 6)
    "eval_capacity" : 2.0,
    "min_capacity" : 4,  # minimum batch size to send to any single expert
    "stride" : 2, # one in every stride layers are converted to an MoE
    "use_switch_tfm_init" : True,  # use weight init scheme from Switch Transformer
    "switch_tfm_init_scale": 1.0,
    "router_use_full_prec" : True  # use float32 precision in the router
}

PARAM_CONFIG_MOE_740M = {
    "vocab_size": 128000, 
    "context_length": 1024, 
    "emb_dim": 768, 
    "num_heads": 12, 
    "num_layers": 16, 
    "dropout_rate": 0.1, 
    "qkv_bias": False, 
    "bias": True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    
    # MoE-related configs 
    "top_k": 3,
    "num_experts": 8,
    "use_aux_loss": True, # apply auxiliary loss (from Switch Transformer) in router
    "use_router_z_loss" : True, # apply router z loss (from ST-MoE)
    "use_noisy_top_k" : True,
    "aux_loss_weight":  0.01, # default setting from Switch Transformer (see top of page 8)
    "router_z_loss_weight":  0.001, # default setting from ST-MoE (see page 8 eq. 6)
    "train_capacity" : 1.25,  # default setting from ST-MoE (see top of page 6)
    "eval_capacity" : 2.0,
    "min_capacity" : 4,  # minimum batch size to send to any single expert
    "stride" : 1, # one in every stride layers are converted to an MoE
    "use_switch_tfm_init" : True,  # use weight init scheme from Switch Transformer
    "switch_tfm_init_scale": 1.0,
    "router_use_full_prec" : True  # use float32 precision in the router
}