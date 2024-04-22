import torch
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2


class LayerNorm(nn.Module):
    """ This is an implementation different from the existing pytorch LayerNorm, with support to set bias= False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_size = config.n_embd
        self.num_heads = config.n_head
        self.head_size = self.embedding_size//self.num_heads
        self.block_size = config.block_size
        
        # print('Block head size : ' ,head_size)
        self.sa = Multihead(num_heads=self.num_heads, 
                            head_size= self.head_size, 
                            embedding_size= self.embedding_size, 
                            block_size= self.block_size)
        self.ffwd = FeedForward(self.embedding_size)
        self.ln1 = LayerNorm(self.embedding_size, bias= config.bias)
        self.ln2 = LayerNorm(self.embedding_size, bias = config.bias)
        
    def forward(self, x):
        x = self.sa(self.ln1(x)) + x
        x = self.ffwd(self.ln2(x)) + x
        return x

class FeedForward(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, 4 *embedding_size),
            nn.ReLU(),
            nn.Linear(4* embedding_size, embedding_size))
        
    def forward(self, x):
        return self.net(x)

class Multihead(nn.Module):
    def __init__(self,num_heads, head_size, embedding_size, block_size):
        super().__init__()  
        self.heads = nn.ModuleList([Head(head_size, embedding_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_size, embedding_size)
        
    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim= -1)
        x = self.proj(x)
        return x
    
class Head(nn.Module):
    def __init__(self, head_size, embedding_size, block_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embedding_size, head_size, bias= False)
        self.query = nn.Linear(embedding_size, head_size, bias= False)
        self.value = nn.Linear(embedding_size, head_size, bias= False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # B,T, head_size
        q = self.query(x) # B,T, head_size
        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5# B,T, head_size @ B,C, head_size => B,T,T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # B,T,T
        wei = F.softmax(wei, dim =-1) #
        
        v =self.value(x)
        out = wei @ v # B,T,T @ B,T, head_size => B,T, head_size
        
        return out

class GPT_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.block_size)
        self.transformer.wte.weight = self.lm_head.weight # weight-tying that pytorch handles itself to transpose 
        
        # initialize the weights
        self.apply(self._init_weights)

    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        Since embeddings are non-trainable, but because of weight-tying (parameter sharing), 
        the Enbedding weights are trained in the final Lm_head. 
        Hence we can remove the parameters from both the embeddings to 
        get the trainable parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params


    def forward(self, idx, targets=None):
        B,T = idx.shape 
        print(B,T)
        device = idx.device
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        idx = torch.tensor([2396,  428,  318,  644,  340, 1724,  284,  492]).to(device= device)        
        print(idx)
        print(idx.shape)
        token_embedding = self.transformer.wte(idx) # (B,T,C)
        print(token_embedding)
        print(token_embedding.shape)
        positional_embedding = self.transformer.wpe(torch.arange(0, T, dtype=torch.long, device= device ))
        
        x = self.transformer.drop(token_embedding + positional_embedding)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        # x = self.sa_head(x)
        # x = self.ffwd(x)
        x  = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
    
    
    
    def configure_optimizers(self, weight_decay, learning_rate, betas):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        return optimizer



    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a sequence of indices idx (b,t) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):

            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / temperature
                
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
