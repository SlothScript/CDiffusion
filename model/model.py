import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Embedding(nn.Module):
    def __init__(self, vocab_size, max_seq_len, hidden_dim, dropout=0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids):
        B, T = token_ids.size()
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_embeddings(token_ids) + self.position_embeddings(positions)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    def __init__(self, hidden_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ff = FeedForward(hidden_dim, ff_dim, dropout)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.resid_dropout(self.attn(self.ln1(x), mask))
        x = x + self.resid_dropout(self.ff(self.ln2(x)))
        return x


class MLMModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len, hidden_dim,
                 num_heads, ff_dim, num_layers=12, dropout=0.1):
        super().__init__()
        self.embeddings = Embedding(vocab_size, max_seq_len, hidden_dim, dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, ff_dim, dropout)
             for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, token_ids, mask=None):
        x = self.embeddings(token_ids)
        for block in self.blocks:
            x = block(x, mask)
        return self.lm_head(self.ln_f(x))


# --- Test Model --- #
if __name__ == "__main__":
    model = MLMModel(
        vocab_size=32768,
        max_seq_len=512,
        hidden_dim=1024,
        num_heads=16,
        ff_dim=4352,
        num_layers=12
    )

    batch_size = 4
    seq_len = 128
    dummy_input = torch.randint(0, 32768, (batch_size, seq_len))

    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output: ({batch_size}, {seq_len}, 32768)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")