import torch
import torch.nn as nn
import torch.nn.functional as F
bertconfig = {
    'vocab_size':30522, # same with bert-base-uncased
    'd_model':768,
    'max_len':256,
    'n_head':12,
    'n_layer':12,
    'dff':4*768,
    'batch_size':1,
    'device':'cuda'
}

class BertFFN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.d_model = config['d_model']
        self.dff = config['dff']

        self.ln1 = nn.Linear(self.d_model, self.dff)
        self.gelu = nn.GELU()
        self.ln2 = nn.Linear(self.dff,self.d_model)

    def forward(self,x:torch.tensor):
        out = self.ln1(x)
        out = self.gelu(out)
        out = self.ln2(out)
        return out

class BertLayerNorm(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(config['d_model']))
        self.shift = nn.Parameter(torch.zeros(config['d_model']))

    def forward(self, x):
        mean =  x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x -mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class BertAttention(nn.Module):

    def __init__(self,config):
        super().__init__()
        
        self.d_model = config['d_model']
        self.n_head = config['n_head']
        self.head_dim = self.d_model // self.n_head
        assert (self.d_model % self.n_head == 0), \
             "emb_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

    def forward(self,x):

        batch_size, seq_len, d_model = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.n_head,self.head_dim)
        k = k.view(batch_size, seq_len, self.n_head,self.head_dim)
        v = v.view(batch_size, seq_len, self.n_head,self.head_dim)
      
        # Diziyi yeniden şekillendiriyoruz (k, q, v sırasıyla): (b, seq_len, n_heads, head_dim) -> (b, n_heads, seq_len, head_dim)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        attn_score = torch.matmul(q,k.transpose(-1,-2)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_score,dim=-1)
        output = torch.matmul(attn_weights, v).transpose(1,2).contiguous()
        output = output.view(batch_size,seq_len,d_model)
        return output
    
class BertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ffn = BertFFN(config)
        self.attn = BertAttention(config)
        self.ln1 = BertLayerNorm(config)
        self.ln2 = BertLayerNorm(config)

    def forward(self, x:torch.tensor):
        shortcut = x
        x = self.attn(x)
        x = self.ln1(x)

        x = x + shortcut

        x = self.ffn(x)
        x = self.ln2(x)

        x = x + shortcut
        return x

class BERT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.device = config['device']   
        self.d_model = config['d_model']
        self.tok_embd_table = nn.Embedding(config['vocab_size'], self.d_model)
        self.pos_embd_table = nn.Embedding(config['max_len'], self.d_model)

        self.encoder_blocks = nn.ModuleList([BertEncoder(config) for _ in range(config['n_layer'])])
        self.linear = nn.Linear(config['d_model'], config['vocab_size'])
    
    def forward(self,x):

        _, seq_len = x.shape

        tok_embd = self.tok_embd_table(x)
        pos_embd = self.pos_embd_table(torch.arange(0,seq_len, device=self.device))
        embd = tok_embd + pos_embd

        for encoder in self.encoder_blocks:
            out = encoder(embd)

        logits = self.linear(out)
        return logits
