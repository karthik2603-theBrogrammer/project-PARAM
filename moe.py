import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from manager import MANAGER


# TODO: Router, Expert, loss functions.

class Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config["top_k"]
        self.n_exp = config["num_experts"]
        assert self.top_k >= 1 and self.top_k <= config["num_experts"]

        self.train_capacity = config["train_capacity"]
        self.eval_capacity = config["eval_capacity"]
        self.min_capacity = config["min_capacity"]
        self.router_use_full_prec = config["router_use_full_prec"]

        # auxiliary / load balancing loss settings
        self.use_aux_loss = config["use_aux_loss"]
        self.use_router_z_loss = config["use_router_z_loss"]
        self.use_noisy_top_k = config["use_noisy_top_k"]


        # linear projection for (noisy) softmax gating
        # no bias is used, see page 4 eq (4) in (https://arxiv.org/abs/1701.06538)
        self.w_g = nn.Linear(config["emb_dim"], config["num_experts"], bias=False)
        self.w_noise = nn.Linear(config["emb_dim"], config["num_experts"], bias=False) if self.use_noisy_top_k else None

    def forward(self, x):
        # perform the router fwd pass in full precision to avoid 
        # numeric instablity (common drawback of MoE's)
        with torch.amp.autocast(device_type = "cuda", enabled = False):
            # optionally run the router in full precision to avoid instability during training
            # see discussion on pg. 9 here: https://arxiv.org/abs/2101.03961
            # setting enabled to False in autocast automatically puts everything in float32
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu' # for later use in torch.autocast
            ctx = nullcontext() if not self.router_use_full_prec else torch.amp.autocast(device_type=device_type, enabled=False)

            with ctx:
                B, T, _ = x.size()
                num_tokens = B * T

                # eq (4) in (https://arxiv.org/abs/1701.06538)
                logits = self.w_g(x)  # [B, T, n_exp]
                if self.use_noisy_top_k:
                    # optionally add noise into the router
                    noise = F.softplus(self.w_noise(x))
                    noise *= torch.randn_like(noise)
                    logits += noise

                # router z loss, computed on logits (before softmax)
                # this loss prevents router logits from becoming too large
                if self.use_router_z_loss:
                    z_loss = self.compute_router_z_loss(logits)
                    MANAGER.add_router_z_loss(z_loss)

                # find top k experts for each token
                top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1) # [B, T, k]

                # Normalize top_k only or all the logits ? According to Shazeer et al (https://arxiv.org/abs/1701.06538) 
                # we do only top_k.
                router_probs = torch.full_like(logits, float('-inf'))  # [B, T, n_exp]
                router_probs.scatter_(-1, top_k_indices, top_k_logits)
                router_probs = F.softmax(router_probs, dim=-1)


                # compute auxiliary load balancing loss
                # this loss encourages equal probability assigned to each expert
                # and equal load balancing of tokens assigned to each expert
                if self.use_aux_loss:
                    aux_loss = self.compute_aux_loss(router_probs, top_k_indices)
                    MANAGER.add_aux_loss(aux_loss)
                
                exp_capacity = self.get_capacity(num_tokens)

                # make a multi-hot mask of chosen experts, size [B, T, n_exp]
                # entries are 0 if expert not chosen and 1 if expert chosen
                exp_mask = F.one_hot(top_k_indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
                exp_mask = exp_mask.view(num_tokens, self.top_k, self.n_exp)  # [B * T, k, n_exp]
                exp_mask = exp_mask.permute(1, 0, 2) # [k, B * T, n_exp]

                # compute cumulative sum of each token over experts, this stores
                # the index of each token within the batch of each expert
                # NOTE: cumsum should count all top-1 first, top-2 second, etc.
                # so that we prioritize top experts when dropping tokens (this is
                # done by putting k dimension first for the reshape operation)
                exp_rank = exp_mask.reshape(self.top_k * num_tokens, self.n_exp)  # [k * B * T, n_exp]
                exp_rank = torch.cumsum(exp_rank, dim=0) - 1  # cumulative sum of expert selections [k * B * T, n_exp]
                exp_rank = exp_rank.reshape(self.top_k, num_tokens, self.n_exp)  # [k, B * T, n_exp]


                # mask out (set to zero) entries that go beyond expert capacity
                # compute amount of used capacity by taking a sum over mask
                exp_mask *= torch.lt(exp_rank, exp_capacity) # [k, B * T, n_exp]
                used_capacity = torch.sum(exp_mask, dim=(0, 1)) # [n_exp]


                # mask rank to only include tokens that are selected
                # perform a sum so each row only contains index of token
                # for the expert that is selected in that row
                # result is a matrix that contains the position of each token
                # in the batch of its corresponding expert
                exp_rank = torch.sum(exp_mask * exp_rank, dim=-1)  # [k, B * T]


                # mask probabilities to only include selected experts
                router_probs = router_probs.view(num_tokens, self.n_exp)[None, :] # [1, B * T, n_exp]
                exp_weights = exp_mask * router_probs # [k, B * T, n_exp]

                # convert rank into one-hot vectors over the available capacity
                # stores the position of each token within the capacity of the selected expert
                exp_rank_sc = F.one_hot(exp_rank, num_classes=exp_capacity) # [k, B * T, exp_capacity]

                # create a vector that stores, for each token, the weight of selected
                # experts at token's position in the capacity of that expert
                # size of tensor is [B * T, n_exp, exp_capacity]
                cb_weight = torch.sum(exp_weights.unsqueeze(3) * exp_rank_sc.unsqueeze(2), dim=0)
                sec_mask = cb_weight.bool() # binary mask of selected experts for each token
                return used_capacity, cb_weight, sec_mask
    
    def compute_aux_loss(self, expert_probs: torch.Tensor, indices: torch.Tensor):
        """
        Computes Switch Transformer auxiliary loss (https://arxiv.org/abs/2101.03961)
        See equations (4)-(6) on page 7
        """

        # equation (5): compute ratio of tokens allocated to each expert
        # total number of tokens is defined as total tokens in batch * k
        # (k = 1) for the Switch Transformer
        with torch.no_grad():
            one_hot_indices = F.one_hot(indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
            one_hot_indices = torch.sum(one_hot_indices.float(), dim=2)  # [B, T, n_exp] (sum over k dimension)
            tokens_per_expert = torch.mean(one_hot_indices.float(), dim=(0, 1))

        # equation (6): compute ratio of router probability allocated to each expert
        prob_per_expert = torch.mean(expert_probs.float(), dim=(0, 1))

        # equation (4): take a scaled dot product between prob/token allocation vectors
        # multiply the result by the number of experts
        return self.n_exp * torch.sum(prob_per_expert * tokens_per_expert)
    
    def compute_router_z_loss(self, logits: torch.Tensor):
        """
        Computes ST-MoE router z loss (https://arxiv.org/abs/2202.08906)
        See equation (5) on page 7
        """
    
        # exponentiate logits, sum logits of each expert, take log, and square
        # code below is the same as:
        # > z_loss = torch.exp(logits)
        # > z_loss = torch.sum(z_loss, dim=-1)
        # > z_loss = torch.log(z_loss) ** 2.0
        z_loss = torch.logsumexp(logits, dim=-1) ** 2.0  # [B, T, n_exp]

        # sum over all tokens and divide by total number of tokens
        return torch.mean(z_loss)
    
    def get_capacity(self, tokens_per_batch):
        # expert capacity is given by (tokens_per_batch / num_experts) * capacity_factor
        # see eq (3) in Switch Transformer (https://arxiv.org/abs/2101.03961)
        capacity_factor = self.train_capacity if self.training else self.eval_capacity
        capacity = math.floor(self.top_k * capacity_factor * tokens_per_batch / self.n_exp)
        capacity += capacity % 2 # make sure capacity is an even number
        capacity = max(capacity, self.min_capacity) # use min capacity
        assert capacity > 0
        return int(capacity)

class MLPExperts(nn.Module):
    """
    implementation of multiple MLP-based experts that can process input
    in batch -- based upon ColossalAI OpenMoE but simple, has optional bias, and
    uses a bmm instead of a loop over a mm for each expert to improve efficiency
    link: https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/moe/experts.py
    """
    def __init__(self, config):
        # TODO: add param init
        super().__init__()
        self.bias = config["bias"]

        self.c_fc = nn.Parameter(torch.empty(config["num_experts"], config["emb_dim"], 4 * config["emb_dim"]))
        self.c_proj = nn.Parameter(torch.empty(config["num_experts"], 4 * config["emb_dim"], config["emb_dim"]))
        self.fc_bias = nn.Parameter(torch.empty(config["num_experts"], 1, 4 * config["emb_dim"])) if self.bias else None
        self.proj_bias = nn.Parameter(torch.empty(config["num_experts"], 1, config["emb_dim"])) if self.bias else None
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config["dropout_rate"])
    

    def forward(self, x):
        x = torch.bmm(x, self.c_fc)
        if self.bias:
            x += self.fc_bias
        x = self.gelu(x)
        x = torch.bmm(x, self.c_proj)
        if self.bias:
            x += self.proj_bias
        x = self.dropout(x)
        return x
    
class MOELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = Router(config) # (noisy) top k router
        self.experts = MLPExperts(config) # group of MLPs (experts)

    def forward(self, x: torch.Tensor):
        B, T, n_embd = x.size() # track original shape of input
        num_tokens = (B * T)

        # pass each token through the router
        used_capacity, exp_weight, exp_mask = self.router(x)

        # flatten out the input
        x = x.view(num_tokens, n_embd)

        # reshape tokens into batches for each expert
        # [n_exp, exp_capacity, B * T] * [B * T, n_embd] -> [n_exp, exp_capacity, n_embd]
        exp_batches = exp_mask.permute(1, 2, 0).type_as(x) @ x

        # compute expert output
        exp_out = self.experts(exp_batches) # [n_exp, exp_capacity, n_embd]

        # aggregate expert outputs based on router weights
        # eq (2) on page 4 of ST-MoE (https://arxiv.org/abs/2202.08906)
        # similar equations are used for other MoE papers
        exp_weight = exp_weight.view(num_tokens, -1) # [B * T, n_exp * exp_capacity]
        exp_out = exp_out.view(-1, n_embd) # [n_exp * exp_capacity, n_embd] 
        output = exp_weight @ exp_out # [B * T, n_embd]
        
        # resize output before return
        return output.view(B, T, n_embd)


# class Expert(nn.Module):
#     """An individual FFN that is activated for a token"""

#     def __init__(self, cfg) -> None:
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
#             GELU(),
#             nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
#         )
    
#     def forward(self, x):
#         return self.layers(x)

# class TopKRouter(nn.Module):
#     def __init__(self, cfg) -> None:
#         super(TopKRouter, self).__init__()
#         self.top_k = cfg['top_k']
#         self.linear = nn.Linear(cfg["emb_dim"], cfg["num_experts"])
#         self.noise_layer = nn.Linear(cfg["emb_dim"], cfg["num_experts"])

#     def forward(self, x):
#         # x is the output from MHA layer.
#         logits = self.linear(x)
#         noise = self.noise_layer(x)

#         noisy_logits = logits + noise

#         top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
#         zeros = torch.full_like(noisy_logits, float('-inf'))
#         sparse_logits = zeros.scatter(-1, indices, top_k_logits)
#         router_output = F.softmax(sparse_logits, dim=-1)
#         return router_output, indices, logits
    
#     def load_balancing_loss_fn(
#             self,
#             router_logits,
#             selected_experts,
#             cfg,
#             attention_mask: Optional[torch.tensor] = None
#     ):
#         if router_logits is None:
#             return torch.tensor(0.0, device = selected_experts.device)
        
#         routing_weights = F.softmax(router_logits, dim = -1)
#         expert_mask = torch.nn.functional.one_hot(selected_experts, cfg["num_experts"])
#         if attention_mask is None:
#             tokens_per_expert = torch.mean(expert_mask.float(), dim=(0,1))
#             router_prob_per_expert = torch.mean(routing_weights, dim=(0, 1))
#         else:
#             batch_size, sequence_length = attention_mask.shape
#             attention_mask = attention_mask.to(selected_experts.device)

#             # Reshape attention mask for expert_mask: [batch, seq_len, 1, 1]
#             expert_attention_mask = attention_mask.unsqueeze(-1).unsqueeze(-1)

#             # Broadcast it to [batch, seq_len, top_k, num_experts]
#             expert_attention_mask = expert_attention_mask.expand(-1, -1, self.top_k, cfg["num_experts"])
        
#             # Apply attention mask to expert mask and compute mean
#             masked_expert_mask = expert_mask.float() * expert_attention_mask
#             tokens_per_expert = torch.sum(masked_expert_mask, dim=(0, 1)) / (
#                 torch.sum(expert_attention_mask, dim=(0, 1)) + 1e-10
#             )

#             # For routing weights, reshape attention mask to [batch, seq_len, 1]
#             router_attention_mask = attention_mask.unsqueeze(-1)

#             # Broadcast to match router_logits shape [batch, seq_len, num_experts]
#             router_attention_mask = router_attention_mask.expand(-1, -1, cfg["num_experts"])

#             # Apply attention mask to routing weights
#             masked_routing_weights = routing_weights * router_attention_mask

#             router_prob_per_expert = torch.sum(masked_routing_weights, dim=(0, 1)) / (
#                 torch.sum(router_attention_mask, dim=(0, 1)) + 1e-10
#             )
        
#         # This is broadcast multiplication
#         # Ex: [top_k, experts] * router_prob_per_expert
#         #     [3, 8] * [8] ==> Each row in top_k is multiplied with the router_probability_per_expert matrix
#         overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert)
#         return overall_loss * cfg["num_experts"]
        



# class SparseMoE(nn.Module):
#     def __init__(self, cfg) -> None:
#         super(SparseMoE, self).__init__()
#         self.cfg = cfg
#         self.router = TopKRouter(cfg= cfg)
#         self.experts = nn.ModuleList([
#             Expert(cfg= cfg) for _ in range(cfg["num_experts"])
#         ])
#         self.top_k = cfg["top_k"]
    
#     def forward(self, x, attention_mask: Optional[torch.tensor] = None):

#         gating_output, indices, router_logits = self.router(x)

#         aux_loss = self.router.load_balancing_loss_fn(
#             router_logits,
#             indices,
#             self.cfg,
#             attention_mask
#         )
#         final_output = torch.zeros_like(x)
#         flat_x = x.view(-1, x.size(-1))
#         flat_gating_output = gating_output.view(-1, gating_output.size(-1))

#         for i, expert in enumerate(self.experts):
#             expert_mask = (indices == i).any(dim=-1)
#             flat_mask = expert_mask.view(-1)

#             if flat_mask.any():
#                 expert_input = flat_x[flat_mask]
#                 expert_output = expert(expert_input)

#                 # Extract and apply gating scores
#                 gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
#                 weighted_output = expert_output * gating_scores

#                 # Update final output additively by indexing and adding
#                 final_output[expert_mask] += weighted_output.squeeze(1)
#         return final_output, aux_loss
