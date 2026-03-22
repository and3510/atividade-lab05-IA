import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model deve ser divisivel por num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        B, S, D = x.shape
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, Q_in, K_in, V_in, mask=None):
        B = Q_in.shape[0]

        Q = self.split_heads(self.W_q(Q_in))
        K = self.split_heads(self.W_k(K_in))
        V = self.split_heads(self.W_v(V_in))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)

        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.W_o(out)

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn       = PositionwiseFFN(d_model, d_ff)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn       = MultiHeadAttention(d_model, num_heads)
        self.ffn              = PositionwiseFFN(d_model, d_ff)
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.norm3  = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, Z, tgt_mask=None, src_mask=None):
        sa_out = self.masked_self_attn(y, y, y, tgt_mask)
        y = self.norm1(y + self.dropout(sa_out))

        ca_out = self.cross_attn(y, Z, Z, src_mask)
        y = self.norm2(y + self.dropout(ca_out))

        ffn_out = self.ffn(y)
        y = self.norm3(y + self.dropout(ffn_out))
        return y

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, y, Z, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            y = layer(y, Z, tgt_mask, src_mask)
        return self.linear(y)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

def make_causal_mask(seq_len):
    mask = torch.tril(torch.ones((seq_len, seq_len)))
    return mask.unsqueeze(0).unsqueeze(0) 

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, d_ff=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding    = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.encoder      = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder      = Decoder(d_model, num_heads, d_ff, num_layers, vocab_size, dropout)
        self.d_model      = d_model

    def encode(self, src, src_mask=None):
        x = self.pos_encoding(self.embedding(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(self, tgt, Z, src_mask=None):
        seq_len  = tgt.size(1)
        tgt_mask = make_causal_mask(seq_len).to(tgt.device)
        y = self.pos_encoding(self.embedding(tgt) * math.sqrt(self.d_model))
        return self.decoder(y, Z, tgt_mask=tgt_mask, src_mask=src_mask)

    def forward(self, src, tgt):
        Z = self.encode(src)
        return self.decode(tgt, Z)