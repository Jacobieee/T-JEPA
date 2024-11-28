import sys

sys.path.append('..')



import torch
import torch.nn as nn
import math
from config import Config


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 201):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000)) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:, token_embedding.size(1), :])


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.attention_head_size * num_heads

        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        query_layer = self.query(query).view(batch_size, -1, self.num_heads, self.attention_head_size).transpose(1, 2)
        key_layer = self.key(key).view(batch_size, -1, self.num_heads, self.attention_head_size).transpose(1, 2)
        value_layer = self.value(value).view(batch_size, -1, self.num_heads, self.attention_head_size).transpose(1, 2)

        # Attention via scaled dot product
        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        if mask is not None:
            # Ensure mask is [batch_size, 1, 1, seq_length] to align with the scores tensor
            mask = mask.unsqueeze(1).unsqueeze(2)  # Add dimensions for heads and compatibility with 'scores'
            # Apply the mask - mask should be broadcastable to the size of scores
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_probs = self.softmax(scores)

        # Context layer calculation
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, -1, self.all_head_size)

        # Final linear projection
        attention_output = self.out(context_layer)

        return attention_output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(embed_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, src, src_mask=None):
        # Self-attention
        src2 = self.multi_head_attention(src, src, src, src_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # Feed-forward
        src2 = self.feed_forward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.traj_embedding = nn.Linear(input_dim, Config.seq_embedding_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.pe = PositionalEncoding(embed_dim, dropout)

    def forward(self, traj_o, src_padding_mask=None):
        src, src_mask = traj_o, src_padding_mask
        src = self.traj_embedding(src)
        src = self.pe(src)
        output = src
        for layer in self.layers:
            output = layer(output, src_mask)
        output = self.norm(output)
        return output


