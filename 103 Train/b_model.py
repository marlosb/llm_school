from dataclasses import dataclass
import inspect
import torch
import torch.nn as nn
import math

# Custom GELU approximation (matches Hugging Face's NewGELUActivation)
class NewGELUActivation(nn.Module):
    def forward(self, x):
# GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        sqrt_2_over_pi = math.sqrt(2 / math.pi)
        cubic_term = 0.044715 * torch.pow(x, 3)
        inner_expr = sqrt_2_over_pi * (x + cubic_term)
        tanh_expr = torch.tanh(inner_expr)
        return 0.5 * x * (1.0 + tanh_expr)

# Attention mechanism
class GPT2Attention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == embed_dim), \
               "embed_dim must be divisible by num_heads"

        # Linear layer for query, key, value 
        # (equivalent to Conv1D with nf=3*embed_dim)
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        
        # Compute query, key, value
        qkv = self.c_attn(x).split(self.embed_dim, dim=-1)
        q, k, v = [
            t.view(batch_size, seq_len, self.num_heads, self.head_dim)
             .transpose(1, 2)
            for t in qkv
        ]  # Shape: [batch, heads, seq_len, head_dim] (more efficient computing)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Output
        out = (torch.matmul(attn, v)
               .transpose(1, 2)
               .contiguous()
               .view(batch_size, seq_len, embed_dim))
        out = self.c_proj(out)
        out = self.resid_dropout(out)
        return out

# Feed-forward network
class GPT2MLP(nn.Module):
    def __init__(self, embed_dim=768, ff_dim=3072, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, ff_dim)
        self.c_proj = nn.Linear(ff_dim, embed_dim)
        self.act = NewGELUActivation()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# Transformer block
class GPT2Block(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.attn = GPT2Attention(embed_dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.mlp = GPT2MLP(embed_dim, ff_dim=3072, dropout=dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x

# DistilGPT-2 model
class DistilGPT2(nn.Module):
    def __init__(self, vocab_size=30000, 
                 max_position=512, 
                 embed_dim=768, 
                 num_layers=6, 
                 num_heads=12, 
                 dropout=0.1, 
                 block_size=512):
        super().__init__()
        
        self.n_layer = num_layers
        self.n_head = num_heads
        self.n_embd = embed_dim 
        self.block_size = block_size
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(vocab_size, embed_dim),
            'wpe': nn.Embedding(max_position, embed_dim),
            'drop': nn.Dropout(dropout),
            'h': nn.ModuleList([
                 GPT2Block(embed_dim, num_heads, dropout) 
                 for _ in range(num_layers)]),
            'ln_f': nn.LayerNorm(embed_dim, eps=1e-5)
        })
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # Tie weights

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
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
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        L, H, Q, T = self.n_layer, self.n_head, self.n_embd//self.n_head, self.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.size()
        max_pos = self.transformer.wpe.num_embeddings
        assert seq_len <= max_pos, \
               f"Sequence length {seq_len} exceeds max_position {max_pos}"
        device = input_ids.device
        
        # Positional indices
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        
        # Embeddings
        token_embeds = self.transformer.wte(input_ids)
        position_embeds = self.transformer.wpe(position_ids)
        x = self.transformer.drop(token_embeds + position_embeds)
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)
        
        # Transformer blocks
        for block in self.transformer.h:
            x = block(x, causal_mask)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)
        
        return logits, loss
    
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 30000 
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster