from __future__ import annotations
from math import log

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear

import einx
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

# orthogonal residual updates
# https://arxiv.org/abs/2505.11881

def orthog_project(x, y):
    dtype = x.dtype

    if x.device.type != 'mps':
        x, y = x.double(), y.double()

    unit = l2norm(y)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthog = x - parallel

    return orthog.to(dtype)

# proposed tanh norm

def tanh_norm(t):
    norm = t.norm(dim = -1, keepdim = True)
    return norm.tanh() * l2norm(t)

# learned scales

class LearnedScale(Module):
    def __init__(
        self,
        dim = 1,
        init_value = 1.,
        rearrange_eq = None
    ):
        super().__init__()
        self.rearrange = Rearrange(rearrange_eq) if exists(rearrange_eq) else nn.Identity()

        log_init_value = log(init_value)
        self.log_scales = nn.Parameter(torch.ones(dim) * log_init_value)

    def forward(self, t):
        scale = self.log_scales.exp()
        scale = self.rearrange(scale)
        return t * scale

# classes

class GatedScreeningTile(Module):
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

        dim_key_value = (dim_keys, dim_values)

        dim_inner = sum(dim_key_value) * heads

        self.to_queries_gates = Linear(dim, dim_inner, bias = False)
        self.to_keys_values = Linear(dim_context, dim_inner, bias = False)

        self.dim_key_value = dim_key_value

        # merging of heads and projecting out

        self.head_wise_scale = LearnedScale(heads, rearrange_eq = 'h -> h 1 1')
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

        queries_gates = self.to_queries_gates(tokens)

        keys_values = self.to_keys_values(key_value_input)

        queries_gates, keys_values = map(self.split_heads, (queries_gates, keys_values))

        # break out the queries, keys, values, gates

        queries, gates = queries_gates.split(self.dim_key_value, dim = -1)
        keys, values = keys_values.split(self.dim_key_value, dim = -1)

        # aggressive normalization

        queries, keys, values = map(l2norm, (queries, keys, values)) # l2norm for queries, keys, and values

        # cosine similarity

        sim = einsum(queries, keys, 'b h i d, b h j d -> b h i j')

        # maybe mask

        if exists(mask):
            sim = einx.where('b j, b h i j,', mask, sim, 0.)

        # relu squared, sans the screening / filtering

        attn = F.relu(sim) ** 2

        # aggregate

        aggr_values = einsum(attn, values, 'b h i j, b h j d -> b h i d')

        # add the proposed tanh norm for further stability

        normed_aggr_values = tanh_norm(aggr_values)

        # gate (24)

        out = normed_aggr_values * F.silu(gates).tanh()

        # author applies headwise scaling, from an old google brain paper iirc

        out = self.head_wise_scale(out)

        # merging and project, sans headwise scaling

        out = self.merge_heads(out)

        return self.to_out(out)

# multiscreen

class MultiScreen(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth = 6,
        **kwargs
    ):
        super().__init__()

        self.token_embeds = nn.Parameter(torch.randn(num_tokens, dim) * 1e-2)

        self.scale_embed = LearnedScale()
        self.scale_unembed = LearnedScale()

        # gated screens

        self.layers = ModuleList([GatedScreeningTile(dim = dim, **kwargs) for _ in range(depth)])

    def forward(self, token_ids):

        # embed

        normed_token_embeds = l2norm(self.token_embeds)
        tokens = normed_token_embeds[token_ids]

        tokens = self.scale_embed(tokens)

        # deep learning

        for layer in self.layers:
            residual = tokens
            block_out = layer(tokens)
            tokens = tokens + orthog_project(block_out, residual)

        # norm

        tokens = l2norm(tokens)

        # unembed

        tokens = self.scale_unembed(tokens)
        logits = einsum(tokens, normed_token_embeds, 'b n d, l d -> b n l')

        return logits
