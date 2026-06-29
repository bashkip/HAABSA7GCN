from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):

    def __init__(self, num_modules: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        if num_modules < 1:
            raise ValueError(f"GatedFusion: num_modules must be >= 1, got {num_modules}")
        self.num_modules = num_modules
        self.hidden_dim = hidden_dim
        self.score_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_modules)
        ])
        self.score_dropout = nn.Dropout(dropout)

    def forward(self, module_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:

        if len(module_outputs) != self.num_modules:
            raise ValueError(
                f"GatedFusion: expected {self.num_modules} module outputs, got {len(module_outputs)}"
            )
        scores = torch.stack(
            [self.score_dropout(W(h)) for W, h in zip(self.score_projections, module_outputs)],
            dim=-1,
        )                                       
        gates = F.softmax(scores, dim=-1)
        stacked = torch.stack(module_outputs, dim=-1)
        fused = (gates * stacked).sum(dim=-1)
        return fused, gates
