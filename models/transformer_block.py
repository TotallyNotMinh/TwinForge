from torch import nn
import torch.nn.functional as F
import torch

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()

        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.dropout_p = dropout

        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.Wo = nn.Linear(dim, dim)

    def forward(self, x, context=None):
        if context is None:
            context = x

        B, N_x, C = x.shape
        N_ctx = context.shape[1]

        q = self.Wq(x).view(B, N_x, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(context).view(B, N_ctx, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(context).view(B, N_ctx, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, N_x, self.dim)
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

        self.norm_context = nn.LayerNorm(normalized_shape=dim)
        self.norm1 = nn.LayerNorm(normalized_shape=dim)
        self.norm2 = nn.LayerNorm(normalized_shape=dim)

        self.mhsa = MultiHeadSelfAttention(dim, num_heads)
        self.ffn = FFN(dim)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x, context=None):
        normed_context = self.norm_context(context) if context is not None else None
        x = x + self.dropout(self.mhsa(self.norm1(x), normed_context))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x # Shape (B, tok_num, dim)

if __name__ == "__main__":
    num_token = 12
    dim = 256
    x = torch.rand((1, num_token, dim))
    sa = TransformerBlock(dim, 8)
    print(torch)
