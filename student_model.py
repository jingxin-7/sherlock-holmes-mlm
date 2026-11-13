"""V1 context reader with an independent clustered head and a tied head."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from v1_model import HEAD_WIDTH, VOCAB_SIZE, StudentModel as V1Model


class TiedHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(HEAD_WIDTH, 128)
        self.norm = nn.LayerNorm(128)
        self.bias = nn.Parameter(torch.zeros(VOCAB_SIZE))

    def forward(self, hidden, embedding):
        z = self.norm(F.gelu(self.projection(hidden)))
        return F.linear(z, embedding, self.bias)


class StudentModel(V1Model):
    def __init__(self):
        super().__init__()
        self.tied = TiedHead()
        self.gate = nn.Linear(HEAD_WIDTH, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(1 / 9))
        self.register_buffer('tied_only', torch.tensor(False))

    @property
    def output_embedding(self):
        # The input embedding module is the only owner of this Parameter.
        return self.bert.embeddings.word_embeddings.weight

    def path_b(self, hidden):
        return self.tied(hidden, self.output_embedding)

    def mix(self, hidden, a, b):
        gate = self.gate(hidden).squeeze(-1).float()
        while gate.ndim < a.ndim:
            gate = gate.unsqueeze(-1)
        return torch.logaddexp(F.logsigmoid(-gate) + a, F.logsigmoid(gate) + b)

    def forward(self, input_ids, attention_mask, mask_positions):
        rows, positions = mask_positions.nonzero(as_tuple=True)
        hidden = self.masked_features(input_ids, attention_mask, rows, positions)
        b = self.path_b(hidden).float()
        if self.tied_only.item():
            return b
        a = self.decoder(hidden).float()
        return self.mix(hidden, a.log_softmax(-1), b.log_softmax(-1))


def load_model(checkpoint_path, device):
    model = StudentModel()
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
