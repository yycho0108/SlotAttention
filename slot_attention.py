#!/usr/bin/env python3

from typing import Callable
from functools import partial
import torch as th
nn = th.nn
F = nn.functional
import einops


# class SlotAttention(nn.Module):
class SlotAttention(th.jit.ScriptModule):
    def __init__(self,
                 n_slots: int,
                 dim_feat: int,
                 n_iter: int = 3,
                 eps: float = 1e-8,
                 dim_hidden: int = 128
                 ):
        super().__init__()
        self.n_slots = n_slots
        self.dim_feat = dim_feat
        self.n_iter = n_iter
        self.eps = eps
        self.dim_hidden = dim_hidden
        self.scale = dim_hidden ** (-0.5)

        self.slots_mu = nn.Parameter(
            nn.init.xavier_uniform_(th.empty(1, 1, dim_feat))
        )  # NOTE(ycho): used to be randn
        self.slots_log_std = nn.Parameter(
            nn.init.xavier_uniform_(th.empty(1, 1, dim_feat))
        )  # NOTE(ycho): used to be zeros

        self.to_q = nn.Linear(dim_feat, dim_feat, bias=False)
        self.to_k = nn.Linear(dim_feat, dim_feat, bias=False)
        self.to_v = nn.Linear(dim_feat, dim_feat, bias=False)
        self.gru = nn.GRUCell(dim_feat, dim_feat)  # ??
        dim_hidden: int = max(dim_feat, dim_hidden)

        self.mlp = nn.Sequential(
            nn.Linear(dim_feat, dim_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(dim_hidden, dim_feat)
        )

        # NOTE(ycho): why LayerNorm?
        self.norm_in = nn.LayerNorm(dim_feat)
        self.norm_slots = nn.LayerNorm(dim_feat)
        self.norm_pre_ff = nn.LayerNorm(dim_feat)  # ??

    @th.jit.script_method
    def _step(self, slots: th.Tensor, k: th.Tensor, v: th.Tensor,
              inplace: bool) -> th.Tensor:
        B, _, D = slots.shape
        prv_slots = slots
        slots = self.norm_slots(slots)
        q = self.to_q(slots)
        dots = th.einsum('bid,bjd->bij', q, k) * self.scale
        attn = th.softmax(dots, dim=1) + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)
        # F.multi_head_attention_forward()?
        updates = th.einsum('bjd,bij->bid', v, attn)
        slots = self.gru(updates.reshape(-1, D),
                         prv_slots.reshape(-1, D)
                         ).reshape(B, -1, D)
        delta = self.mlp(self.norm_pre_ff(slots))
        if inplace:
            slots += delta
        else:
            slots = slots + delta
        return slots

    @th.jit.script_method
    def forward(self, x: th.Tensor):
        B, N, D = x.shape
        S = self.n_slots

        mu = self.slots_mu.expand(B, S, -1)
        sigma = self.slots_log_std.exp().expand(B, S, -1)
        slots = mu + sigma * th.randn_like(mu)

        x = self.norm_in(x)

        k, v = self.to_k(x), self.to_v(x)

        # NOTE(ycho): version with implicit differentiation.
        with th.no_grad():
            for _ in range(self.n_iter):
                slots = self._step(slots, k, v, True)
        slots = self._step(slots.detach(), k, v, False)
        return slots


def main():
    sa = SlotAttention(4, 128, 8)
    x = th.randn(size=(2, 16, 128))
    y = sa(x)
    print('y', y.shape)  # (2, 4, 128)


if __name__ == '__main__':
    main()
