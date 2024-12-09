import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert (
            self.head_dim * num_heads == dim
        ), "Embedding dimension must be divisible by number of heads"

        self.scale = self.head_dim ** -0.5
        
        # Linear layers for queries, keys, and values
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        
        # Output linear layer
        self.out_linear = nn.Linear(dim, dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.q_linear(q)  # (batch_size, seq_len, dim)
        k = self.k_linear(k)  # (batch_size, seq_len, dim)
        v = self.v_linear(v)  # (batch_size, seq_len, dim)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # Compute attention scores
        attention_scores = (q @ k.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_len, seq_len)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = torch.softmax(attention_scores, dim=-1)

        # Apply attention scores to the values
        attention_output = attention_scores @ v  # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads and pass through the output linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)  # (batch_size, seq_len, dim)
        return self.out_linear(attention_output), attention_scores