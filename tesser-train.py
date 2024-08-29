import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import GradScaler, autocast

class TesseractEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_planes=24):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_planes = num_planes
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.init_plane_embeddings()

    def init_plane_embeddings(self):
        # Dynamic initialization based on embedding dimension
        self.plane_embeddings = nn.Parameter(torch.randn(self.num_planes, self.embed_dim) / math.sqrt(self.embed_dim))

    def forward(self, x):
        base_embed = self.embedding(x)
        plane_embeds = base_embed.unsqueeze(1) + self.plane_embeddings.unsqueeze(0).unsqueeze(0)
        return plane_embeds

class TesseractPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_length, num_planes=24):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        self.num_planes = num_planes
        
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_seq_length, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # 4D rotations for each plane (shared parameters for efficiency)
        num_shared_rotations = 4
        self.plane_rotations = nn.Parameter(torch.randn(num_shared_rotations, 4, 4))
        self.plane_rotation_assignment = nn.Parameter(torch.randint(0, num_shared_rotations, (num_planes,)))
        
    def forward(self, x):
        b, t, p, e = x.size()
        pe = self.pe[:t].unsqueeze(0).unsqueeze(2).expand(b, -1, p, -1)
        rotated_pe = torch.einsum('btpe,pij->btpje', pe.view(b, t, p, 4, -1), 
                                  self.plane_rotations[self.plane_rotation_assignment])
        return x + rotated_pe.view(b, t, p, e)

class TesseractMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_planes=24, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_planes = num_planes
        
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self.plane_attention = nn.Parameter(torch.randn(num_planes, num_planes))
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        
        # Learnable scaling factor for attention weights
        self.learnable_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x, mask=None):
        b, t, p, e = x.size()
        
        q = self.q_proj(x).view(b, t, p, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        k = self.k_proj(x).view(b, t, p, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v = self.v_proj(x).view(b, t, p, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        
        attn = torch.einsum('bhtpe,bhtqe->bhpqt', q, k) * self.scale
        
        # Apply learnable scaling factor
        attn = attn * self.learnable_scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = self.attn_dropout(F.softmax(attn, dim=-1))
        
        out = torch.einsum('bhpqt,bhtqe->bhpte', attn, v)
        out = out.permute(0, 2, 3, 1, 4).contiguous().view(b, t, p, e)
        
        return self.out_dropout(self.o_proj(out.view(b, t, -1)).view(b, t, p, e))

class TesseractFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(self.gelu(self.fc1(x)))))

class TesseractTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_planes=24, dropout=0.1):
        super().__init__()
        self.attn = TesseractMultiHeadAttention(embed_dim, num_heads, num_planes, dropout)
        self.ff = TesseractFeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Pre-activation residual connections
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

class TesseractTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_length, num_planes=24, dropout=0.1):
        super().__init__()
        self.embedding = TesseractEmbedding(vocab_size, embed_dim, num_planes)
        self.pos_encoding = TesseractPositionalEncoding(embed_dim, max_seq_length, num_planes)
        self.layers = nn.ModuleList([
            TesseractTransformerLayer(embed_dim, num_heads, ff_dim, num_planes, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim * num_planes, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        x = self.dropout(self.embedding(x))
        x = self.dropout(self.pos_encoding(x))
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.fc(x)
        
        return x

# Training and evaluation functions with enhancements
def train_step(model, optimizer, criterion, x, y, scaler):
    model.train()
    optimizer.zero_grad()
    with autocast():
        output = model(x)
        loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()

def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        with autocast():
            output = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            preds = output.argmax(dim=-1)
            accuracy = (preds == y).float().mean()
    return loss.item(), accuracy.item()

# Hyperparameters
vocab_size = 50000
embed_dim = 512
num_heads = 8
ff_dim = 2048
num_layers = 12
max_seq_length = 1024
num_planes = 24
dropout = 0.1
lr = 1e-4
warmup_steps = 1000
epochs = 10
batch_size = 32
grad_clip = 1.0

# Initialize model
model = TesseractTransformer(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_length, num_planes, dropout)

# Optimizer, scheduler, and scaler
optimizer = Adam(model.parameters(), lr=lr)
scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0))
scaler = GradScaler()

# Loss function
criterion = nn.CrossEntropyLoss()

# Training loop (pseudo-code, replace with actual data loading)
best_eval_loss = float('inf')
early_stop_counter = 0
patience = 3

for epoch in range(epochs):
    for batch in range(num_batches):
        x, y = load_batch(batch_size, max_seq_length)
        loss = train_step(model, optimizer, criterion, x, y, scaler)
        print(f"Epoch {epoch}, Batch {batch},
