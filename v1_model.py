"""Encoder-only Tiny with a relative gap reader and hierarchical decoder."""
import math

import torch
from torch import nn
from torch.nn import functional as F

VOCAB_SIZE = 30522
MASK_ID = 103
CLUSTER_SIZES = (2048, 6144, VOCAB_SIZE - 8192)
HEAD_WIDTH = 192


class Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(VOCAB_SIZE, 128, padding_idx=0)
        self.position_embeddings = nn.Embedding(512, 128)
        self.token_type_embeddings = nn.Embedding(2, 128)
        self.LayerNorm = nn.LayerNorm(128, eps=1e-12)
        self.dropout = nn.Dropout(.08)
    def forward(self, ids):
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        return self.dropout(self.LayerNorm(self.word_embeddings(ids) +
            self.position_embeddings(pos) + self.token_type_embeddings(torch.zeros_like(ids))))

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(128,128)
        self.key = nn.Linear(128,128)
        self.value = nn.Linear(128,128)
    def forward(self, x, mask):
        b,l,_ = x.shape
        q,k,v = [f(x).view(b,l,2,64).transpose(1,2) for f in (self.query,self.key,self.value)]
        y = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,
            dropout_p=.08 if self.training else 0.)
        return y.transpose(1,2).contiguous().view(b,l,128)

class ResidualOutput(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.dense = nn.Linear(width,128)
        self.LayerNorm = nn.LayerNorm(128,eps=1e-12)
        self.dropout = nn.Dropout(.08)
    def forward(self,x,residual):
        return self.LayerNorm(self.dropout(self.dense(x))+residual)

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.self = SelfAttention()
        self.output = ResidualOutput(128)
    def forward(self,x,mask):
        return self.output(self.self(x,mask),x)

class Intermediate(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(128,512)
    def forward(self,x):
        return F.gelu(self.dense(x))

class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention()
        self.intermediate = Intermediate()
        self.output = ResidualOutput(512)
    def forward(self,x,mask):
        x = self.attention(x,mask)
        return self.output(self.intermediate(x),x)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([Layer(), Layer()])

    def forward(self, x, mask):
        first = self.layer[0](x, mask)
        return first, self.layer[1](first, mask)


class Bert(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = Embeddings()
        self.encoder = Encoder()

    def forward(self, ids, attention_mask):
        visible = attention_mask[:, None, None, :].bool()
        return self.encoder(self.embeddings(ids), visible)


class GapReader(nn.Module):
    """Mix immediate left/right syntax with contextual retrieval at each gap."""

    def __init__(self):
        super().__init__()
        self.skip = nn.Linear(128, HEAD_WIDTH)
        self.query = nn.Linear(128, 64)
        self.key = nn.Linear(128, 64)
        self.value = nn.Linear(128, 128)
        self.distance_bias = nn.Parameter(torch.zeros(4, 33))
        self.neighbor = nn.Linear(4 * 128, 128)
        self.fuse = nn.Sequential(
            nn.LayerNorm(128 * 3), nn.Linear(128 * 3, 256),
            nn.GELU(), nn.Linear(256, HEAD_WIDTH),
        )
        self.norm = nn.LayerNorm(HEAD_WIDTH)
        nn.init.normal_(self.fuse[-1].weight, std=0.005)
        self.register_buffer('neighbor_offsets', torch.tensor([-2, -1, 1, 2]), persistent=False)

    def forward(self, first, last, attention_mask, rows, positions):
        length = last.shape[1]
        center = last[rows, positions]
        query = self.query(center).reshape(-1, 4, 16)
        keys = self.key(first).reshape(-1, length, 4, 16)[rows].transpose(1, 2)
        values = self.value(last).reshape(-1, length, 4, 32)[rows].transpose(1, 2)
        offsets = torch.arange(length, device=last.device)[None, :] - positions[:, None]
        scores = (query[:, :, None] * keys).sum(-1) * 0.25
        scores = scores + self.distance_bias[:, (offsets + 16).clamp(0, 32)].transpose(0, 1)
        scores = scores.masked_fill(~attention_mask[rows, None, :].bool(), -torch.inf)
        context = (scores.softmax(-1)[..., None] * values).sum(2).flatten(1)

        neighbors = positions[:, None] + self.neighbor_offsets
        valid = neighbors.ge(0) & neighbors.lt(length)
        neighbors = neighbors.clamp(0, length - 1)
        valid = valid & attention_mask[rows[:, None], neighbors].bool()
        local = first[rows[:, None], neighbors] * valid[..., None]
        local = self.neighbor(local.flatten(1))
        delta = self.fuse(torch.cat([first[rows, positions], local, context], dim=-1))
        return self.norm(self.skip(center) + delta)


class ClusterDecoder(nn.Module):
    """Exact hierarchical CE with conditional classifiers for two tail bands."""

    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(HEAD_WIDTH, width, bias=False) for width in (128, 96, 64)])
        self.root = nn.Linear(128, CLUSTER_SIZES[0] + 2)
        self.tails = nn.ModuleList(
            [nn.Linear(width, size) for width, size in zip((96, 64), CLUSTER_SIZES[1:])]
        )
        self.register_buffer('vocab_to_rank', torch.arange(VOCAB_SIZE))
        self.register_buffer('backoff_log_probs', torch.full((VOCAB_SIZE,), -torch.inf))
        self.register_buffer('log_main_mass', torch.tensor(0.0))
        self.dropout = nn.Dropout(0.12)

    def features(self, hidden, band):
        return self.projections[band](hidden)

    def loss(self, hidden, root_labels, tails):
        hidden = self.dropout(hidden)
        root = self.root(self.features(hidden, 0))
        loss = F.cross_entropy(root.float(), root_labels, reduction='sum')
        for band, (rows, labels) in enumerate(tails, 1):
            if labels.numel():
                scores = self.tails[band - 1](self.features(hidden[rows], band))
                loss = loss + F.cross_entropy(scores.float(), labels, reduction='sum')
        return loss / root_labels.numel()

    def forward(self, hidden):
        root = self.root(self.features(hidden, 0))
        common = CLUSTER_SIZES[0]
        bands = [root[:, :common]]
        for i, classifier in enumerate(self.tails, 1):
            scores = classifier(self.features(hidden, i))
            bands.append(root[:, common + i - 1:common + i] + F.log_softmax(scores, dim=-1))
        logits = torch.cat(bands, dim=1)[:, self.vocab_to_rank]
        # Keep root's arbitrary logit offset when combining with unseen-token mass.
        return torch.logaddexp(
            logits + self.log_main_mass, root.logsumexp(-1, keepdim=True) + self.backoff_log_probs
        )


class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = Bert()
        self.gap = GapReader()
        self.decoder = ClusterDecoder()

    def masked_features(self, ids, attention_mask, rows, positions):
        first, last = self.bert(ids, attention_mask)
        return self.gap(first, last, attention_mask, rows, positions)

    def forward(self, input_ids, attention_mask, mask_positions):
        rows, positions = mask_positions.nonzero(as_tuple=True)
        hidden = self.masked_features(input_ids, attention_mask, rows, positions)
        return self.decoder(hidden)


def load_model(checkpoint_path, device):
    model = StudentModel()
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True), strict=True)
    return model.to(device).eval()
