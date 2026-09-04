from torch import nn
import torch

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        attention_matrix = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        attention = self.dropout(self.softmax(attention_matrix))
        return torch.matmul(attention, value)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()

        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.Wo = nn.Linear(dim, dim)
        
        self.attention = SelfAttention()

    def forward(self, x):
        B, N, C = x.shape

        q = self.Wq(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2) # (B, N_token, n_heads, dim) -> (B, n_heads, N_token, dim), batch matrix multiplication operates on the last two dimensions (N, token, dim)
        k = self.Wk(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        out = self.attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.dim)
        return self.Wo(out)


class FFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W1 = nn.Linear(dim, dim * 2)
        self.W2 = nn.Linear(dim * 2, dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.W2(self.gelu(self.W1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        
        self.num_heads = num_heads
        self.dim = dim

        self.norm1 = nn.LayerNorm(normalized_shape=dim)
        self.norm2 = nn.LayerNorm(normalized_shape=dim)

        self.mhsa = MultiHeadSelfAttention(dim, num_heads)
        self.ffn = FFN(dim)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x + self.dropout(self.mhsa(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x # Shape (B, tok_num, dim)

if __name__ == "__main__":
    num_token = 12
    dim = 256
    x = torch.rand((1, num_token, dim))
    sa = TransformerBlock(dim, 8)
    print(torch)
