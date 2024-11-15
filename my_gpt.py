import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import logging 

block_size = 128
vocab_size = 500
n_embed = 384
dropout = 0.2
n_head = 6
n_layer = 6
kv_heads = 3
max_position_embeddings = 128

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
    

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads, head_dim) :
        super().__init__()
        assert num_heads%kv_heads == 0
        self.n_embed = n_embed
        self.num_attn_heads = num_heads
        self.head_dim = head_dim
        self.kv_heads = kv_heads
        # self.kv_out_proj = head_dim *  self.kv_heads #Doubt
        self.num_kv_groups = self.num_attn_heads // self.kv_heads


        self.heads = nn.ModuleList(Head(head_size=head_dim) for _ in range(num_heads))
        ##Only self attention

        #For num_attn_heads number of heads
        self.Wq = nn.Linear(self.n_embed, self.num_attn_heads*self.head_dim)
        #For kv_heads number of heads
        self.Wk = nn.Linear(self.n_embed, self.kv_heads * self.head_dim)
        self.Wv = nn.Linear(self.n_embed, self.kv_heads * self.head_dim)

        self.o_proj = nn.Linear(self.head_dim * self.num_attn_heads, self.n_embed)
        self.dropout = nn.Dropout(dropout)

        # self.attention_mask = torch.zeros((bsz, self.num_attn_heads, qlen, qlen))
        # self.attention_mask[:, :, :, qlen:] = float('-inf')  # Mask out positions beyond the key sequence length


    def forward(self, x, attn_mask= None):
        """
        Parameters:
            x (bsz, qlen, embed) : input
        """
        # out = torch.cat([h(x) for h in self.heads], dim=-1)
        # attn_output = self.dropout(self.o_proj(out))

        # ################ Experiment


        bsz, qlen, embed = x.size()

        q = self.Wq(x) ##(B,T,head_dim * num_heads)
        k = self.Wk(x) ##(B,T,head_dim * kv_heads)
        v = self.Wv(x) ##(B,T,head_dim * kv_heads)



        q = q.view(bsz, qlen, self.num_attn_heads, self.head_dim).transpose(2,1)  ##(B,T,head_dim * num_heads)
        k = k.view(bsz, qlen, self.kv_heads, self.head_dim).transpose(2,1) ##(B,T,head_dim * kv_heads)
        v = v.view(bsz, qlen, self.kv_heads, self.head_dim).transpose(2,1)  ##(B,T,head_dim * kv_heads)

        # print("k-shape before ",k.shape)
        k = repeat_kv(k, self.num_kv_groups) ##(B, n_kvheads * nrep, qlen, head_dim)
        v = repeat_kv(v, self.num_kv_groups)

        attn_scores = q @ k.transpose(-1,-2)/torch.sqrt(torch.tensor(self.n_embed)) ##(B, T, block_size)

        ################
        
        if attn_mask is not None:
            # causal_mask = attn_mask[:, :, :, : k.shape[-2]]
            # attn_scores = attn_scores + causal_mask
            attn_scores = attn_scores.masked_fill(
                attn_mask[None, None, :qlen, :qlen]==0 , float("-inf")
            )


        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = F.dropout(attn_scores) ##Why this dropout is required??

        attn_output = torch.matmul(attn_scores, v) ##(B, n_heads, qlen, hidden_size)
        attn_output = attn_output.transpose(1,2).contiguous()    
        attn_output = attn_output.view(bsz, qlen, self.n_embed)

        attn_output = self.o_proj(attn_output)
        return attn_output


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
    def __init__(self, n_embed, n_heads, attn_mask=None):
        super().__init__()
        # Assume 0 for allowed positions and -inf for masked positions
        
        self.sa = MultiHeadAttention(n_heads,n_embed//n_head)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed)
        self.register_buffer('causal_mask',torch.tril(torch.ones(block_size,block_size)))



    def forward(self, x):
        x = x + self.sa(self.ln1(x), attn_mask = self.causal_mask)
        x = x + self.ffwd(self.ln2(x))
        return x



class my_gpt(nn.Module):
    def __init__(self, device="cpu", block_size = 256):
        super().__init__()
        self.device = device
        self.block_size = block_size ##context window size
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.pos_embed = nn.Embedding(max_position_embeddings, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.sa_head = Head(vocab_size)
        self.d_blocks = nn.Sequential(*[decoder_block(n_embed=n_embed,n_heads=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets = None):
        """
        Args:
            x: int(B,T) Token ids
            targets :

        Returns:
            logits
        """
        B, T = x.size() ##
        tok_emd = self.token_embed(x) ##(B,T,C)
        position_ids = torch.arange(T, device = self.device )
        pos_emd = self.pos_embed(position_ids)

        
        x = tok_emd + pos_emd

        x = self.d_blocks(x) #
        x = self.ln_f(x) # (B,T,C)

        logits = self.lm_head(x) ##(B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
            
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
        for _ in range(max_new_tokens): 
            ##Take only last allowed number of tokens
            idx_tokens = context[:, -self.block_size:]

            ##generate the next token
            logits, loss = self(idx_tokens)

            ##Take only last allowed number of tokens
            logits = logits[:,-1,:] ##(B,vocab_size)

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

    



