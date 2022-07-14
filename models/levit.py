from math import ceil

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from cuda.shift import Shift
# helpers
def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)


class tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class Shift_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(Shift_tcn, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        bn_init(self.bn2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.shift_in = Shift(channel=in_channels, stride=1, init_scale=1)
        self.shift_out = Shift(channel=out_channels, stride=1, init_scale=1)

        self.temporal_linear = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal(self.temporal_linear.weight, mode='fan_out')

    def forward(self, x):
        x = self.bn(x)
        # shift1
        x = self.shift_in(x)
        x = self.temporal_linear(x)
        x = self.relu(x)
        # shift2
        x = self.shift_out(x)
        x = self.bn2(x)
        return x
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, l = 3):
    val = val if isinstance(val, tuple) else (val,)
    return (*val, *((val[-1],) * max(l - len(val), 0)))

def always(val):
    return lambda *args, **kwargs: val

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, mult, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, fmap_size, heads = 8, dim_key = 32, dim_value = 64, dropout = 0., dim_out = None, downsample = False):
        super().__init__()
        inner_dim_key = dim_key *  heads
        inner_dim_value = dim_value *  heads
        dim_out = default(dim_out, dim)

        self.heads = heads
        self.scale = dim_key ** -0.5

        self.to_q = nn.Sequential(nn.Conv2d(dim, inner_dim_key, 1, stride = (2 if downsample else 1), bias = False), nn.BatchNorm2d(inner_dim_key))
        self.to_k = nn.Sequential(nn.Conv2d(dim, inner_dim_key, 1, bias = False), nn.BatchNorm2d(inner_dim_key))
        self.to_v = nn.Sequential(nn.Conv2d(dim, inner_dim_value, 1, bias = False), nn.BatchNorm2d(inner_dim_value))

        self.attend = nn.Softmax(dim = -1)

        out_batch_norm = nn.BatchNorm2d(dim_out)
        nn.init.zeros_(out_batch_norm.weight)

        self.to_out = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inner_dim_value, dim_out, 1),
            out_batch_norm,
            nn.Dropout(dropout)
        )

        # positional bias

        self.pos_bias = nn.Embedding(fmap_size * fmap_size, heads)

        q_range = torch.arange(0, fmap_size, step = (2 if downsample else 1))
        k_range = torch.arange(fmap_size)

        q_pos = torch.stack(torch.meshgrid(q_range, q_range), dim = -1)
        k_pos = torch.stack(torch.meshgrid(k_range, k_range), dim = -1)

        q_pos, k_pos = map(lambda t: rearrange(t, 'i j c -> (i j) c'), (q_pos, k_pos))
        rel_pos = (q_pos[:, None, ...] - k_pos[None, :, ...]).abs()

        x_rel, y_rel = rel_pos.unbind(dim = -1)
        pos_indices = (x_rel * fmap_size) + y_rel

        self.register_buffer('pos_indices', pos_indices)

    def apply_pos_bias(self, fmap):
        bias = self.pos_bias(self.pos_indices)
        bias = rearrange(bias, 'i j h -> () h i j')
        return fmap + (bias / self.scale)

    def forward(self, x):
        b, n, *_, h = *x.shape, self.heads

        q = self.to_q(x)
        y = q.shape[2]

        qkv = (q, self.to_k(x), self.to_v(x))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        dots = self.apply_pos_bias(dots)

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, fmap_size, depth, heads, dim_key, dim_value, mlp_mult = 2, dropout = 0., dim_out = None, downsample = False):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.layers = nn.ModuleList([])
        self.attn_residual = (not downsample) and dim == dim_out

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, fmap_size = fmap_size, heads = heads, dim_key = dim_key, dim_value = dim_value, dropout = dropout, downsample = downsample, dim_out = dim_out),
                FeedForward(dim_out, mlp_mult, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            attn_res = (x if self.attn_residual else 0)
            x = attn(x) + attn_res
            x = ff(x) + x
        return x

class LeViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_mult,
        stages = 3,
        dim_key = 32,
        dim_value = 64,
        dropout = 0.,
        num_distill_classes = None
    ):
        super().__init__()

        dims = cast_tuple(dim, stages)
        depths = cast_tuple(depth, stages)
        layer_heads = cast_tuple(heads, stages)

        assert all(map(lambda t: len(t) == stages, (dims, depths, layer_heads))), 'dimensions, depths, and heads must be a tuple that is less than the designated number of stages'
        """
        self.conv_embedding = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride = 2, padding = 1),
            nn.Conv2d(32, 64, 3, stride = 2, padding = 1),
            nn.Conv2d(64, 128, 3, stride = 2, padding = 1),
            nn.Conv2d(128, dims[0], 3, stride = 2, padding = 1)
        )
        """
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride = 2,padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride = 2,padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride = 2,padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(128, dims[0], 3, stride = 2,padding = 1),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(dims[0]*2, dims[0], 3, stride = 1,padding = 1),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU()
        )

        self.conv_embedding = nn.Sequential(
            self.conv_1,
            self.conv_2,
            self.conv_3,
            self.conv_4
        )


        self.tcn_1 = Shift_tcn(3, 32)
        self.tcn_2 = Shift_tcn(32, 64)
        self.tcn_3 = Shift_tcn(64, 128)
        self.tcn_4 = Shift_tcn(128, dims[0])
        self.conv_1y = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride = 2,padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_2y = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride = 2,padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv_3y = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride = 2,padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_4y = nn.Sequential(
            nn.Conv2d(128, dims[0], 3, stride = 2,padding = 1),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU()
        )
        self.conv_embedding = nn.Sequential(
            self.conv_1,
            self.conv_2,
            self.conv_3,
            self.conv_4
        )


        self.tcn_1y = Shift_tcn(3, 32)
        self.tcn_2y = Shift_tcn(32, 64)
        self.tcn_3y = Shift_tcn(64, 128)
        self.tcn_4y = Shift_tcn(128, dims[0])


        fmap_size = image_size // (2 ** 4)
        layers = []

        for ind, dim, depth, heads in zip(range(stages), dims, depths, layer_heads):
            is_last = ind == (stages - 1)
            layers.append(Transformer(dim, fmap_size, depth, heads, dim_key, dim_value, mlp_mult, dropout))

            if not is_last:
                next_dim = dims[ind + 1]
                layers.append(Transformer(dim, fmap_size, 1, heads * 2, dim_key, dim_value, dim_out = next_dim, downsample = True))
                fmap_size = ceil(fmap_size / 2)

        self.backbone = nn.Sequential(*layers)

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...')
        )

        self.distill_head = nn.Linear(dim, num_distill_classes) if exists(num_distill_classes) else always(None)
        self.mlp_head = nn.Linear(dim, 512)

    def forward(self, img, img_fft):
        #x = self.conv_embedding(img)
        x=img
        x_tcn_1 = self.tcn_1(x)
        x = self.conv_1(x)
        #print(x.size())
        x = x + x_tcn_1
        x_tcn_2 = self.tcn_2(x)
        x = self.conv_2(x)
        x = x + x_tcn_2
        x_tcn_3 = self.tcn_3(x)
        x = self.conv_3(x)
        x = x + x_tcn_3
        x = self.conv_4(x)
        
        y=img_fft
        y_tcn_1 = self.tcn_1y(y)
        y = self.conv_1y(y)
        #print(x.size())
        y = y + y_tcn_1
        y_tcn_2 = self.tcn_2y(y)
        y = self.conv_2y(y)
        y = y + y_tcn_2
        y_tcn_3 = self.tcn_3y(y)
        y = self.conv_3y(y)
        y = y + y_tcn_3
        y = self.conv_4y(y)
        #print(x.size())
        x = self.conv(torch.cat([x,y],dim=1))
        #print(x.size())



        x = self.backbone(x)        

        x = self.pool(x)

        out = self.mlp_head(x)
        distill = self.distill_head(x)

        if exists(distill):
            return out, distill

        return out
