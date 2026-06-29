from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


VALID_SPLITS = ("train", "val", "test")


class XCatLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int, embedding_dim: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.dense = nn.Linear(embedding_dim, in_features, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, X: torch.Tensor, r: torch.Tensor) -> torch.Tensor:

        N = X.size(0)
        d_in = X.size(-1)
        inv_sqrt = d_in ** -0.5

        # Centre row attention: softmax over the centre's neighbours (0..K).
        x0 = X[0]
        scores0 = (X @ x0) * inv_sqrt                  # (1+K,)
        attn0 = F.softmax(scores0, dim=0)               # (1+K,)

        # hidden[j] = X[j] @ weight ; the (W_r r) @ W term is shared (added at the end)
        hidden = X @ self.weight                        # (1+K, out_features)
        r_proj_out = self.dense(r) @ self.weight        # (out_features,)

        out0 = (attn0.unsqueeze(-1) * hidden).sum(dim=0, keepdim=True)   # (1, out_features)

        if N > 1:
            # Non-centre rows: each has only two edges (to centre, self-loop).
            x_rest = X[1:]                              # (K, d_in)
            score_ic = (x_rest * x0).sum(dim=-1) * inv_sqrt    # (K,)
            score_ii = (x_rest * x_rest).sum(dim=-1) * inv_sqrt  # (K,)
            scores_rest = torch.stack([score_ic, score_ii], dim=-1)   # (K, 2)
            attn_rest = F.softmax(scores_rest, dim=-1)                 # (K, 2)
            # out_rest[i] = attn[i, 0] * hidden[0] + attn[i, 1] * hidden[i+1]
            out_rest = attn_rest[:, 0:1] * hidden[0:1] + attn_rest[:, 1:2] * hidden[1:]
            out = torch.cat([out0, out_rest], dim=0)
        else:
            out = out0

        out = out + r_proj_out
        if self.bias is not None:
            out = out + self.bias
        return out


class CrossExampleGraphs:

    def __init__(self, year: int, data_dir: Path | str):
        path = Path(data_dir) / f"cross_features_{year}.pt"
        payload = torch.load(str(path), weights_only=False)
        self.year = year
        self.bert_model: str = payload.get("bert_model", "")
        self.pooling: str = payload.get("pooling", "")
        self.category_vocab: Dict[str, int] = payload["category_vocab"]
        self.num_categories: int = len(self.category_vocab)

        self.train_X: torch.Tensor = payload["train"]["X"]
        self.train_categories: torch.Tensor = payload["train"]["categories"]
        self.train_aspects: List[str] = payload["train"]["aspect_terms"]

        self.test_X: torch.Tensor = payload["test"]["X"]
        self.test_categories: torch.Tensor = payload["test"]["categories"]
        self.test_aspects: List[str] = payload["test"]["aspect_terms"]

        self.hidden_size: int = int(self.train_X.size(-1))

        self._cat_to_train_idx: List[torch.Tensor] = [
            (self.train_categories == c).nonzero(as_tuple=True)[0]
            for c in range(self.num_categories)
        ]
        self._cat_to_full_train_idx: List[torch.Tensor] = [t.clone() for t in self._cat_to_train_idx]

        inv = {v: k for k, v in self.category_vocab.items()}
        sparse = [(c, int(idx.numel())) for c, idx in enumerate(self._cat_to_train_idx) if idx.numel() < 2]
        if sparse:
            for c, n in sparse:
                print(f"[CrossExampleGraphs] WARNING category {inv[c]!r} (id={c}) has {n} train mentions; "
                      f"XCat star reduces to self-loop only for train mentions of this category")

        self.xsim_top_k = None
        self._xsim_sim_train_raw = None
        self._xsim_full_train_topk = None
        self._xsim_train_topk = None
        self._xsim_full_test_topk = None

    # ----- accessors -----

    def split_X(self, split: str) -> torch.Tensor:
        if split == "train":
            return self.train_X
        if split == "test":
            return self.test_X
        raise ValueError(f"unknown split {split!r}")

    def split_categories(self, split: str) -> torch.Tensor:
        if split == "train":
            return self.train_categories
        if split == "test":
            return self.test_categories
        raise ValueError(f"unknown split {split!r}")

    def category_id(self, split: str, mention_id: int) -> int:
        return int(self.split_categories(split)[mention_id].item())

    def xcat_neighbours(self, split: str, mention_id: int) -> torch.Tensor:

        if split not in VALID_SPLITS:
            raise ValueError(f"unknown split {split!r}")
        if split == "test":
            cat = self.category_id("test", mention_id)
            return self._cat_to_full_train_idx[cat]
        # train or val centre: both are train-XML mentions, so look up category
        # in the train table. Pool is the filtered true-train pool.
        cat = self.category_id("train", mention_id)
        pool = self._cat_to_train_idx[cat]
        if split == "train":
            pool = pool[pool != mention_id]
        return pool


    def set_train_subset(self, train_indices) -> None:

        if hasattr(train_indices, 'tolist'):
            train_indices = train_indices.tolist()
        keep = set(int(i) for i in train_indices)
        dev = self._cat_to_full_train_idx[0].device if self.num_categories else torch.device('cpu')
        self._cat_to_train_idx = [
            torch.tensor(
                [int(i) for i in self._cat_to_full_train_idx[c].tolist() if int(i) in keep],
                dtype=torch.long, device=dev,
            )
            for c in range(self.num_categories)
        ]

        # If XSim is enabled, rebuild the train-centre top-K filtered
        if self.xsim_top_k is not None:
            keep_mask = torch.zeros(self.train_X.size(0), dtype=torch.bool, device=dev)
            keep_mask[torch.tensor(sorted(keep), dtype=torch.long, device=dev)] = True
            sim_filtered = self._xsim_sim_train_raw.clone()
            sim_filtered.fill_diagonal_(-float('inf'))      # exclude self
            sim_filtered[:, ~keep_mask] = -float('inf')      # exclude val cols
            K_eff = min(self.xsim_top_k, int(keep_mask.sum().item()))
            if K_eff < 1:
                raise RuntimeError('set_train_subset: filtered XSim pool is empty')
            self._xsim_train_topk = sim_filtered.topk(K_eff, dim=-1).indices

    def set_xsim_top_k(self, top_k: int) -> None:

        if int(top_k) < 1:
            raise ValueError(f"xsim_top_k must be >= 1, got {top_k}")
        self.xsim_top_k = int(top_k)

        # Cache the raw (M_train, M_train) cosine matrix so set_train_subset()
        # can rebuild the filtered top-K without re-running the matmul.
        Xt = F.normalize(self.train_X, dim=-1)
        self._xsim_sim_train_raw = Xt @ Xt.T

        # Full-pool train-centre top-K (train ∪ val cols; self-excluded via diag).
        sim_for_full = self._xsim_sim_train_raw.clone()
        sim_for_full.fill_diagonal_(-float('inf'))
        K_eff = min(self.xsim_top_k, sim_for_full.size(-1) - 1)
        self._xsim_full_train_topk = sim_for_full.topk(K_eff, dim=-1).indices
        # Until set_train_subset is called, train-centre lookups use the full table.
        self._xsim_train_topk = self._xsim_full_train_topk

        # Test-centre full-pool top-K: test mentions vs all train mentions.
        # No self-exclusion: test mentions are never in train_X.
        Xte = F.normalize(self.test_X, dim=-1)
        sim_te = Xte @ Xt.T
        K_eff_te = min(self.xsim_top_k, sim_te.size(-1))
        self._xsim_full_test_topk = sim_te.topk(K_eff_te, dim=-1).indices

    def xsim_neighbours(self, split: str, mention_id: int) -> torch.Tensor:
        if self.xsim_top_k is None:
            raise RuntimeError(
                'xsim_neighbours called but set_xsim_top_k() was never invoked. '
                'The Instructor should call set_xsim_top_k(opt.xsim_top_k) before training '
                'whenever opt.modules[xsimgcn] is True.'
            )
        if split not in VALID_SPLITS:
            raise ValueError(f'unknown split {split!r}')
        if split == 'test':
            return self._xsim_full_test_topk[mention_id]
        # train or val centre: both index into the train-rows table.
        return self._xsim_train_topk[mention_id]

    # ----- device management (regular Python object, manual .to) -----

    def to(self, device: torch.device | str) -> "CrossExampleGraphs":
        device = torch.device(device) if not isinstance(device, torch.device) else device
        self.train_X = self.train_X.to(device)
        self.train_categories = self.train_categories.to(device)
        self.test_X = self.test_X.to(device)
        self.test_categories = self.test_categories.to(device)
        self._cat_to_train_idx = [t.to(device) for t in self._cat_to_train_idx]
        self._cat_to_full_train_idx = [t.to(device) for t in self._cat_to_full_train_idx]
        # move XSim tensors if they have been built.
        if self._xsim_sim_train_raw is not None:
            self._xsim_sim_train_raw = self._xsim_sim_train_raw.to(device)
            self._xsim_full_train_topk = self._xsim_full_train_topk.to(device)
            self._xsim_train_topk = self._xsim_train_topk.to(device)
            self._xsim_full_test_topk = self._xsim_full_test_topk.to(device)
        return self

    def describe(self) -> str:
        inv = {v: k for k, v in self.category_vocab.items()}
        lines = [
            f"CrossExampleGraphs(year={self.year}, bert_model={self.bert_model}, pooling={self.pooling})",
            f"  hidden_size={self.hidden_size}  num_categories={self.num_categories}",
            f"  train: M={len(self.train_aspects)}  X={tuple(self.train_X.shape)}",
            f"  test : M={len(self.test_aspects)}  X={tuple(self.test_X.shape)}",
            "  per-category train counts:",
        ]
        for c in range(self.num_categories):
            lines.append(f"    [{c:2d}] {inv[c]:30s} {int(self._cat_to_train_idx[c].numel()):4d}")
        return "\n".join(lines)
