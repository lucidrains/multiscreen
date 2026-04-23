from __future__ import annotations
from typing import Callable
from math import log, ceil

import torch
from torch import nn, pi, arange, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear

import einx
from einops import einsum, rearrange
from einops.layers.torch import Rearrange

from PoPE_pytorch import PoPE, apply_pope_to_qk

from discrete_continuous_embed_readout import ParameterlessReadout

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def inv_sqrt(x):
    return x ** -0.5

# tensor helpers

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def l1norm(t):
    return F.normalize(t, p = 1, dim = -1)

def init_normal_(t, scale = 0.1):
    dim = t.shape[-1]
    nn.init.normal_(t, std = scale * inv_sqrt(dim))

# simple sampling filter

def top_k(logits, frac_num_tokens = 0.1, k = None):
    num_tokens = logits.shape[-1]

    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(k, num_tokens)

    value, indices = logits.topk(k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, indices, value)
    return probs

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
        bias = 0.,
        rearrange_eq = None
    ):
        super().__init__()
        self.rearrange = Rearrange(rearrange_eq) if exists(rearrange_eq) else nn.Identity()

        log_init_value = log(init_value)
        self.log_scales = nn.Parameter(torch.ones(dim) * log_init_value)

        self.bias = bias

    def forward(self, t = None):
        scale = self.log_scales.exp()

        scale = self.rearrange(scale) + self.bias

        if not exists(t):
            return scale

        return t * scale

# distance aware soft mask

class SoftMask(Module):
    def __init__(
        self,
        heads,
        window_threshold = 256
    ):
        super().__init__()
        self.heads = heads
        self.window_threshold = window_threshold
        self.learned_window = LearnedScale(heads, bias = 1., rearrange_eq = 'h -> h 1 1')

    @torch.no_grad()
    def init_(self):
        log_scales_init = torch.linspace(0., log(self.window_threshold), self.heads)
        self.learned_window.log_scales.copy_(log_scales_init)

    def forward(
        self,
        sim
    ):
        i, j, device = *sim.shape[2:], sim.device
        offset = j - i

        assert i <= j

        # learned window

        w = self.learned_window()

        # get distance

        seq_i, seq_j = tuple(arange(n, dtype = torch.float, device = device) for n in (i, j))

        distance = einx.subtract('j, i -> i j', seq_j, seq_i + offset)

        # eq (17)

        soft_mask = torch.where(
            (distance <= 0.) & (distance > -w),
            0.5 * (torch.cos((distance * pi) / w) + 1.),
            0.
        )

        return sim * soft_mask

# classes

class GatedScreeningTile(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_context = None,
        dim_keys = 16,
        dim_values = 64,
        use_pope = True,
        dim_pope = 8,
        causal = True,
        distance_aware_soft_mask = True,
        depth_for_init = 6,
        window_threshold = 256,
        competitive = False
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_keys = dim_keys
        self.dim_values = dim_values
        self.competitive = competitive
        dim_context = default(dim_context, dim)

        # relative positions
        # author overly concerned with length extrap, seems unaware of recent work https://arxiv.org/abs/2509.10534
        # we will just use partial PoPE here

        assert dim_pope <= dim_keys
        self.pope = PoPE(dim = dim_pope, heads = heads) if use_pope else None

        # distance aware soft mask

        self.soft_mask = SoftMask(heads, window_threshold = window_threshold) if distance_aware_soft_mask else None

        # autoregressive or not

        self.causal = causal

        # queries, keys, values

        dim_key_value = (dim_keys, dim_values)

        dim_inner = sum(dim_key_value) * heads

        self.to_queries_gates = Linear(dim, dim_inner, bias = False)
        self.to_keys_values = Linear(dim_context, dim_inner, bias = False)

        self.dim_key_value = dim_key_value

        # merging of heads and projecting out

        self.to_out = Linear(dim_values * heads, dim, bias = False)

        # learned parameters for screening and headwise scale at end

        self.inverse_acceptance_width = LearnedScale(heads, bias = 1., rearrange_eq = 'h -> h 1 1') # r in paper, as r increases, you filter / screen out less
        self.head_wise_scale = LearnedScale(heads, rearrange_eq = 'h -> h 1 1', init_value = inv_sqrt(heads * depth_for_init))

        # split and merging of heads

        self.split_heads = Rearrange('... n (h d) -> ... h n d', h = heads)
        self.merge_heads = Rearrange('... h n d -> ... n (h d)')

        # init

        self.init_()

    @torch.no_grad()
    def init_(self):
        heads = self.heads

        q_w, g_w = rearrange(self.to_queries_gates.weight, '(h o) ... -> h o ...', h = heads).split(self.dim_key_value, dim = 1)
        k_w, v_w = rearrange(self.to_keys_values.weight, '(h o) ... -> h o ...', h = heads).split(self.dim_key_value, dim = 1)

        init_normal_(q_w)
        init_normal_(k_w)
        init_normal_(v_w)
        init_normal_(self.to_out.weight)

        nn.init.normal_(g_w, std = 0.1)

        if exists(self.soft_mask):
            self.soft_mask.init_()

    def forward(
        self,
        tokens,
        context = None,
        mask = None,
        pos_emb = None,
        apply_pos_emb: Callable | None = None
    ):
        # support cross attention

        key_value_input = default(context, tokens)

        # maybe pope

        if exists(self.pope):
            seq_len = max((key_value_input.shape[-2], tokens.shape[-2]))
            pos_emb = self.pope(seq_len)

        # queries, keys, values

        queries_gates = self.to_queries_gates(tokens)

        keys_values = self.to_keys_values(key_value_input)

        queries_gates, keys_values = map(self.split_heads, (queries_gates, keys_values))

        # break out the queries, keys, values, gates

        queries, gates = queries_gates.split(self.dim_key_value, dim = -1)
        keys, values = keys_values.split(self.dim_key_value, dim = -1)

        # aggressive normalization

        queries, keys, values = map(l2norm, (queries, keys, values)) # l2norm for queries, keys, and values

        # maybe rotate

        if exists(self.pope):
            queries, keys = self.pope.apply_pope_to_qk(pos_emb, queries, keys)

        elif exists(pos_emb):
            assert exists(apply_pos_emb), f'`apply_pos_emb` function must be passed in, for your rotary or polar positional embeddings'
            queries, keys = apply_pos_emb(pos_emb, queries, keys)

        # cosine similarity

        sim = einsum(queries, keys, 'b h i d, b h j d -> b h i j')

        # content screening

        # in the paper, r >= 1 and 1/r is the acceptance width.

        r = self.inverse_acceptance_width()

        screened_sim = 1. - r * (1. - sim) # eq (16)

        # relu squared

        attn = F.relu(screened_sim) ** 2

        # maybe mask

        if exists(mask):
            attn = einx.where('b j, b h i j,', mask, attn, 0.)

        if self.soft_mask:
            attn = self.soft_mask(attn)
        elif self.causal:
            i, j = attn.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = sim.device).triu(j - i + 1)
            attn = attn.masked_fill(causal_mask, 0.)

        # maybe competitive

        if self.competitive:
            attn = l1norm(attn)

        # aggregate

        aggr_values = einsum(attn, values, 'b h i j, b h j d -> b h i d')

        # add the proposed tanh norm for further stability

        normed_aggr_values = tanh_norm(aggr_values)

        # gate (24)

        out = normed_aggr_values * F.silu(gates).tanh()

        # author applies headwise scaling, from an old google brain paper iirc

        out = self.head_wise_scale(out)

        # merging and project

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
        heads = 8,
        dim_pope = 4,
        dim_keys = 16,
        dim_values = 64,
        competitive: bool | tuple[bool, ...] = False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim

        self.token_embeds = nn.Parameter(torch.randn(num_tokens, dim) * 1e-2)

        self.scale_embed = LearnedScale()
        self.scale_unembed = LearnedScale(init_value = dim ** 0.5)

        # relative positions with pope
        # https://arxiv.org/abs/2509.10534

        self.pope = PoPE(dim = dim_pope, heads = heads)

        # per-layer competitive flags

        if isinstance(competitive, bool):
            competitive = (competitive,) * depth

        assert len(competitive) == depth, f'competitive tuple length {len(competitive)} must match depth {depth}'

        # gated screens

        self.layers = ModuleList([GatedScreeningTile(
            dim = dim,
            heads = heads,
            causal = True,
            depth_for_init = depth,
            dim_pope = dim_pope,
            dim_keys = dim_keys,
            dim_values = dim_values,
            use_pope = False,
            competitive = layer_competitive,
            **kwargs
        ) for layer_competitive in competitive])

        # readout

        self.readout = ParameterlessReadout(num_discrete = 1)

        # init

        self.init_()

    @torch.no_grad()
    def init_(self):
        init_normal_(self.token_embeds)

        for layer in self.layers:
            layer.init_()

    @torch.no_grad()
    def generate(
        self,
        prompt,
        seq_len,
        temperature = 1.,
        filter_fn = top_k,
        filter_kwargs: dict = dict()
    ):
        prompt_len = prompt.shape[-1]

        assert seq_len > prompt_len
        generate_len = seq_len - prompt_len

        out = prompt

        for _ in range(generate_len):
            logits = self.forward(out)

            last_logits = logits[:, -1:]
            filtered_logits = filter_fn(last_logits, **filter_kwargs)

            sampled = self.readout.sample(filtered_logits, temperature = temperature)

            out = cat((out, sampled), dim = -1)

        return out[..., prompt_len:]

    def forward(
        self,
        token_ids,
        return_loss = False
    ):

        # returning autoregressive loss

        if return_loss:
            token_ids, labels = token_ids[..., :-1], token_ids[..., 1:]

        # embed

        normed_token_embeds = l2norm(self.token_embeds)
        tokens = normed_token_embeds[token_ids]

        tokens = self.scale_embed(tokens)

        # positions

        seq_len = token_ids.shape[-1]
        pos_emb = self.pope(seq_len)

        # deep learning

        for layer in self.layers:
            block_out = layer(
                tokens,
                pos_emb = pos_emb,
                apply_pos_emb = apply_pope_to_qk
            )

            tokens = tokens + orthog_project(block_out, tokens)

        # norm

        tokens = l2norm(tokens)

        # unembed

        tokens = self.scale_unembed(tokens)
        logits = einsum(tokens, normed_token_embeds, 'b n d, l d -> b n l')

        # returning logits only

        if not return_loss:
            return logits

        loss = F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            labels
        )

        return loss
