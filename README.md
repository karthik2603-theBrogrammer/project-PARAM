# Project-PARAM

A suite of MoE based language models developed purely for learning reasons only. 

## Model Architecture

- PARAM follows a decoder-only architecture inspired by the original GPT architecture.
- A sample PARAM model architecture is depicted below. Every alternate hidden layer has a MoE layer and a naive FFN respectively.
- The flow is as follows:
  ```
  EMBEDDING → DROPOUT → N × TRANSFORMER BLOCK 
            (DROPOUT → Multi-Head Attention → DROPOUT → LayerNorm → MoE / FFN)
            → FINAL NORM → OUTPUT LM HEAD

  ```

```
ParamModel(
  (token_emb): Embedding(128002, 768)
  (position_emb): Embedding(1024, 768)
  (dropout_emb): Dropout(p=0.1, inplace=False)
  (transformer_blocks): ModuleList(
    (0): TransformerBlock(
      (norm1): LayerNorm()
      (mhaunit): MultiHeadAttention(
        (dropout): Dropout(p=0.1, inplace=False)
        (W_query): Linear(in_features=768, out_features=768, bias=False)
        (W_key): Linear(in_features=768, out_features=768, bias=False)
        (W_value): Linear(in_features=768, out_features=768, bias=False)
        (out_proj): Linear(in_features=768, out_features=768, bias=True)
      )
      (dropout): Dropout(p=0.1, inplace=False)
      (norm2): LayerNorm()
      (moe): MOELayer(
        (router): Router(
          (w_g): Linear(in_features=768, out_features=8, bias=False)
          (w_noise): Linear(in_features=768, out_features=8, bias=False)
        )
        (experts): MLPExperts(
          (gelu): GELU(approximate='none')
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (1): TransformerBlock(
      (norm1): LayerNorm()
      (mhaunit): MultiHeadAttention(
        (dropout): Dropout(p=0.1, inplace=False)
        (W_query): Linear(in_features=768, out_features=768, bias=False)
        (W_key): Linear(in_features=768, out_features=768, bias=False)
        (W_value): Linear(in_features=768, out_features=768, bias=False)
        (out_proj): Linear(in_features=768, out_features=768, bias=True)
      )
      (dropout): Dropout(p=0.1, inplace=False)
      (norm2): LayerNorm()
      (moe): FeedForwardNetwork(
        (layers): Sequential(
          (0): Linear(in_features=768, out_features=3072, bias=True)
          (1): GELU()
          (2): Linear(in_features=3072, out_features=768, bias=True)
        )
      )
    )
  )
  (final_norm): LayerNorm()
  (out_head): Linear(in_features=768, out_features=128002, bias=False)
)
```

## Training

- PARAM-0.5B has been trained for `271,380` steps for several days. The loss curve has been indicated as below. 
- Datasets Used: wikipedia + stackexchange open-source datasets have been used.

<img width="887" height="390" alt="Screenshot 2025-11-04 at 5 15 31 PM" src="https://github.com/user-attachments/assets/4ee3b1d2-de12-4b32-bb17-d061a7f5b50a" />


## Weights
- The weights are currently in the process of being uploaded to HuggingFace. This page will be updated once done.


## References and Contact

- [Mixture-of-Experts (MoE): The Birth and Rise of Conditional Computation](https://cameronrwolfe.substack.com/p/conditional-computation-the-birth)
- [Build a Large Language Model (From Scratch) ](https://www.manning.com/books/build-a-large-language-model-from-scratch)

**Contact**:  Karthik Namboori  (namkarthik2003@gmail.com)

For collaborations, feature requests, or bug reports, please reach out.

  
