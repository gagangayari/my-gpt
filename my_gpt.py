import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import logging 

with open('config.json', 'r') as file:
    config = json.load(file)

block_size = 256
vocab_size = 65
n_embed = 512
dropout = 0.2
n_head = 32
n_layer = 24

class Head(nn.Module):
    def __init__(self, head_size=16):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)
       


    def forward(self,x):
        B,T,C = x.shape

        q = self.query(x)
        k = self.key(x)

        wei = (q @ k.transpose(-2,-1)) * (k.shape[-1]**(-0.5))
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)

        out = wei @ v ## (B,T,HS)

        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads, head_size) :
        super().__init__()

        self.heads = nn.ModuleList(Head(head_size=head_size) for _ in range(num_heads))
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self,n_embed) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,4* n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.net(x)
        return x

class decoder_block(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        self.sa = MultiHeadAttention(n_heads,n_embed//n_heads)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



class my_gpt(nn.Module):
    def __init__(self, block_size = 256):
        super().__init__()
        self.block_size = block_size ##context window size
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.pos_embed = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.sa_head = Head(vocab_size)
        self.d_blocks = nn.Sequential(*[decoder_block(n_embed=n_embed,n_heads=n_head) for _ in range(n_layer)])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets = None):
        """
        Args:
            idx: int(B,T) Token ids
            targets :

        Returns:
            logits
        """
        # print("idx ", idx)
        B, T = idx.shape ##
        tok_emd = self.token_embed(idx) ##(B,T,C)
        pos_emd = self.pos_embed(idx)

        
        x = tok_emd + pos_emd
        # print("x1 ", x.shape)

        x = self.d_blocks(x) #

        logits = self.lm_head(x) ##(B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # print("logits ", logits.shape)
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
        # print("Logits", logits.shape)
            
        return logits, loss


    def generate(self, context : torch.tensor, max_new_tokens: int = 46, use_cache = False):
        """
        Generates the next "max_new_tokens" number of tokens.

        Args:
            context (B,T):
            max_new_tokens (int):

        Returns:
            [token] : List of generated tokens.
        """
        # print("Context:" , context)
        for _ in range(max_new_tokens): 
            ##Take only last allowed number of tokens
            idx_tokens = context[:, -self.block_size:]

            ##generate the next token
            logits, loss = self(idx_tokens)

            ##Take only last allowed number of tokens
            logits = logits[:,-1,:] ##(B,vocab_size)
            # print("logits:" , logits.shape)

            probs = F.softmax(logits, dim= -1)
            idx_next = torch.multinomial(probs,num_samples=1) ##(B,1)
   
            context = torch.concatenate([context, idx_next], dim=1)

        return context
    
    def save_pretrained(self, path):
        torch.save(self.state_dict(),path)
        print("Saved pretrained Successfully")

    @classmethod
    def load_pretrained(cls, path):
        print("Loading pretrained model...")
        model = cls()
        model.load_state_dict(torch.load(path))
        return  model

    



