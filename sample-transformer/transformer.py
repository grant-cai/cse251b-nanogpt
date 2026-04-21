import torch
import math
from torch import nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.Q = nn.Linear(n_embd, n_embd)
        self.K = nn.Linear(n_embd, n_embd)
        self.V = nn.Linear(n_embd, n_embd)
        self.output = nn.Linear(n_embd, n_embd)

    def forward(self, x, mask=None):
        batch, seq_len, n_embd = x.shape
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        Q = Q.contiguous().view(batch, seq_len, self.n_head, self.head_dim)
        Q = Q.transpose(1,2)

        K = K.contiguous().view(batch, seq_len, self.n_head, self.head_dim)
        K = K.transpose(1,2)

        V = V.contiguous().view(batch, seq_len, self.n_head, self.head_dim)
        V = V.transpose(1,2)

        K_T = K.transpose(-2,-1)

        # Calculate slope for each head for AliBi
        head_indices = torch.arange(1, self.n_head+1, device=x.device)
        slopes = 2**(-8 / self.n_head * head_indices)
        slopes = -slopes
        slopes = slopes.view(1,self.n_head, 1, 1)

        i = torch.arange(seq_len, device=x.device)
        j = torch.arange(seq_len, device=x.device)
        i = i.view(-1,1)
        j = j.view(1,-1)
        distances = i - j
        distances = torch.abs(distances)
        alibi_bias = slopes * distances

        attn_scores = torch.matmul(Q,K_T) / math.sqrt(self.head_dim)
        attn_scores = attn_scores + alibi_bias # add alibi bias
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)

        x = torch.matmul(attn_weights, V)
        x = x.transpose(1,2)
        x = x.contiguous().view(batch, seq_len, n_embd)
        x = self.output(x)

        return x, attn_weights.mean(dim=1)

class FeedForward(nn.Module):
    def __init__(self, n_embd, n_hidden):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_embd)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_hidden):
        super().__init__()
        self.attention = MultiHeadAttention(n_embd, n_head)
        self.ffn = FeedForward(n_embd, n_hidden)
        self.l1 = nn.LayerNorm(n_embd)
        self.l2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        normalized = self.l1(x)
        attn_out, attn_weights = self.attention(normalized)
        x = x + attn_out

        normalized = self.l2(x)
        ff_out = self.ffn(normalized)
        x = x + ff_out
        
        return x, attn_weights


class Encoder(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_hidden, n_layer):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            EncoderBlock(n_embd, n_head, n_hidden) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        token_emb = self.token_embedding(x)
        positions = torch.arange(seq_len, device = x.device)
        # pos_emb = self.position_embedding(positions)

        x = token_emb # + pos_emb

        attn_maps = []
        for block in self.blocks:
            x, attn = block(x)
            attn_maps.append(attn)
        
        x = self.ln_f(x)

        return x, attn_maps

class Classifier(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class JointClassifier(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
    
    def forward(self, x):
        embeddings,_ = self.encoder(x)
        mask = (x != 0).unsqueeze(-1).float()
        masked_embeddings = embeddings * mask
        sum_embeddings = masked_embeddings.sum(dim=1)
        lengths = mask.sum(dim=1)
        pooled = sum_embeddings / lengths
        logits = self.classifier(pooled)
        return logits

class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_hidden):
        super().__init__()
        self.attention = MultiHeadAttention(n_embd, n_head)
        self.ffn = FeedForward(n_embd, n_hidden)
        self.l1 = nn.LayerNorm(n_embd)
        self.l2 = nn.LayerNorm(n_embd)
    
    def forward(self, x, mask):
        normalized = self.l1(x)
        attn_out, attn_weights = self.attention(normalized, mask)
        x = x + attn_out

        normalized = self.l2(x)
        ff_out = self.ffn(normalized)
        x = x + ff_out
        
        return x, attn_weights
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_hidden, n_layer):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            DecoderBlock(n_embd, n_head, n_hidden) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x, targets=None):
        batch_size, seq_len = x.shape
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        token_emb = self.token_embedding(x)
        positions = torch.arange(seq_len, device = x.device)
        # pos_emb = self.position_embedding(positions)

        x = token_emb # + pos_emb

        attn_maps = []
        for block in self.blocks:
            x, attn = block(x, mask)
            attn_maps.append(attn)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return loss
        else:
            return logits, attn_maps