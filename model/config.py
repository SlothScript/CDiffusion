from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

model_type = "test" # test, 225m, 2b

if model_type == "test":
    # Small model (~50M params)
    vocab_size = 32768
    hidden_dim = 512
    num_layers = 8
    num_heads = 8
    ffn_dim = 2048
    max_seq_len = 512
    dropout = 0.1

elif model_type == "225m":
    # Medium model (~225M params)
    vocab_size = 32768
    hidden_dim = 1024
    num_layers = 12
    num_heads = 16
    ffn_dim = 4352
    max_seq_len = 512
    dropout = 0.1

elif model_type == "2b":
    # Large model (~2B params)
    vocab_size = 32768
    hidden_dim = 2560
    num_layers = 24
    num_heads = 32
    ffn_dim = 10240
    max_seq_len = 1024
    dropout = 0.1

# Training
batch_size = 32
learning_rate = 1e-4
warmup_steps = 10_000
max_steps = 100_000
save_every = 5_000

optimizer_class = AdamW
optimizer_kwargs = dict(
    lr=learning_rate,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)

scheduler_class = CosineAnnealingLR
scheduler_kwargs = dict(
    T_max=max_steps - warmup_steps,
    eta_min=1e-6,
)

# Data
mask_rate_min = 0.15
mask_rate_max = 0.30