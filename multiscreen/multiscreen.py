from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, ModuleList

from einops import einsum
from einops.layers.torch import Rearrange

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# tensor helpers

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

# proposed tanh norm

def tanh_norm(t):
    norm = t.norm(dim = -1, keepdim = True)
    return norm.tanh() * l2norm(t)

# classes

class MultiScreen(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_context = None,
        dim_keys = 16,
        dim_values = 64
    ):
        super().__init__()
        dim_context = default(dim_context, dim)

        # queries, keys, values

        dim_key_values = (dim_keys, dim_values)

        self.to_queries = Linear(dim, dim_keys * heads, bias = False)
        self.to_keys_values = Linear(dim_context, sum(dim_key_values) * heads, bias = False)

        self.dim_key_values = dim_key_values

        # merging of heads and projecting out

        self.to_out = Linear(dim_values * heads, dim, bias = False)

        # split and merging of heads

        self.split_heads = Rearrange('... n (h d) -> ... h n d', h = heads)
        self.merge_heads = Rearrange('... h n d -> ... n (h d)')

    def forward(
        self,
        tokens,
        context = None,
        mask = None
    ):
        # support cross attention

        key_value_input = default(context, tokens)

        # queries, keys, values

        queries = self.to_queries(tokens)

        keys_values = self.to_keys_values(key_value_input)

        queries, keys_values = map(self.split_heads, (queries, keys_values))

        keys, values = keys_values.split(self.dim_key_values, dim = -1)

        # aggressive normalization

        queries, keys, values = map(l2norm, (queries, keys, values)) # l2norm for queries, keys, and values

        # cosine similarity

        sim = einsum(queries, keys, 'b h i d, b h j d -> b h i j')

        # relu squared, sans the screening / filtering

        attn = F.relu(sim) ** 2

        # aggregate

        out = einsum(attn, values, 'b h i j, b h j d -> b h i d')

        # add the proposed tanh norm for further stability

        out = tanh_norm(out)

        # merging and project, sans headwise scaling

        out = self.merge_heads(out)

        return self.to_out(out)
