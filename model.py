#!/usr/bin/env python3

from typing import Callable, Tuple, Iterable
from functools import partial
import torch as th
nn = th.nn
F = nn.functional
import einops

from slot_attention import SlotAttention
from common import PositionEmbedding, Conv2D, ConvT2D


class SlotAttentionEncoder(nn.Module):
    def __init__(self, dim_out: int, kernel_size: int = 5,
                 padding: int = 2):
        super().__init__()
        h = dim_out
        k = kernel_size
        p = padding
        self.layer = nn.Sequential(
            Conv2D(3, h, k, p),
            Conv2D(h, h, k, p),
            Conv2D(h, h, k, p),
            Conv2D(h, h, k, p),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.layer(x)


class SlotAttentionDecoder(nn.Module):
    def __init__(self, dim_hidden: int, kernel_size: int, padding: int,
                 stride: int, output_dim, out_kernel_size: int, out_padding: int):
        super().__init__()
        self.layer = nn.Sequential(
            # Upconv 4x
            ConvT2D(dim_hidden, dim_hidden, stride, kernel_size, padding),
            ConvT2D(dim_hidden, dim_hidden, stride, kernel_size, padding),
            ConvT2D(dim_hidden, dim_hidden, stride, kernel_size, padding),
            ConvT2D(dim_hidden, dim_hidden, stride, kernel_size, padding),
            # Last two layers are sort of special, for some reason.
            Conv2D(dim_hidden, dim_hidden, 5, 2),
            nn.Conv2d(dim_hidden, output_dim, 3, 1, 1)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.layer(x)


class SlotAttentionAE(nn.Module):
    def __init__(self,
                 resolution: Tuple[int, int] = (128, 128),
                 n_slots: int = 8,
                 dim_hidden: int = 64,
                 n_iter: int = 3):
        super().__init__()
        self.n_slots = n_slots
        self.slot_attention = SlotAttention(n_slots, dim_hidden, n_iter)
        self.encoder_cnn = SlotAttentionEncoder(dim_hidden)
        self.encoder_pos = PositionEmbedding(dim_hidden, resolution)
        self.decoder_pos = PositionEmbedding(dim_hidden, (8, 8))
        # NOTE(ycho): output_dim = 4 = (3 (img) + 1 (mask))
        self.decoder_cnn = SlotAttentionDecoder(dim_hidden, 4, 1, 2,
                                                4, 4, 1)
        self.decoder_initial_size: Tuple[int, int] = (8, 8)
        self.layer_norm = nn.LayerNorm(dim_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden)
        )

    def forward(
            self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        x0 = x

        x = self.encoder_cnn(x)  # B C H W
        x = self.encoder_pos(x)  # B C H W
        x = einops.rearrange(x, '... d h w -> ... (h w) d')  # B (H W) D'
        x = self.mlp(self.layer_norm(x))  # B (H W) D'

        slots = self.slot_attention(x)  # B (S) D
        batch_dims = slots.shape[:-2]

        # NOTE(ycho): sorry, what the hell is the point of repeat() here??
        x = einops.repeat(slots, '... s d -> (... s) d h w',
                          h=self.decoder_initial_size[0],
                          w=self.decoder_initial_size[1])  # (BXS) D H W
        # debug_memory()
        x = self.decoder_pos(x)
        x = self.decoder_cnn(x)
        x = x.reshape(batch_dims + (self.n_slots,) + x.shape[-3:])
        rec = x[..., :-1, :, :]  # ... num_slots num_chans h w
        msk = x[..., -1, :, :]  # ... num_slots h w
        msk = F.softmax(msk, dim=-3)
        out = th.einsum('... s d h w, ... s h w -> ... d h w', rec, msk)

        # if x.shape[-2:] != x0.shape[-2:]
        # x = F.interpolate(x, x0.shape[-2:])
        return (out, rec, msk, slots)


# def debug_memory():
#     import collections
#     import gc
#     import resource
#     import torch
#     print('maxrss = {}'.format(
#         resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
#     tensors = collections.Counter(
#         (str(o.device), str(o.dtype), tuple(o.shape))
#         for o in gc.get_objects()
#         if torch.is_tensor(o)
#     )
#     for line in sorted(tensors.items()):
#         print('{}\t{}'.format(*line))


def main():
    device = th.device('cuda')
    resolution: Tuple[int, int] = (128, 128)
    model = SlotAttentionAE(resolution).to(device)
    img = th.zeros((2, 3,) + resolution, device=device)
    out = model(img)
    print(out[0].shape)


if __name__ == '__main__':
    main()
