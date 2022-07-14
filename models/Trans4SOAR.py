# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

# Modified from
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# Copyright 2020 Ross Wightman, Apache-2.0 License
from ast import JoinedStr
from copy import deepcopy
from ntpath import join
import torch.nn.functional as F
import torch
import itertools
import utils
import numpy as np
from timm.models.vision_transformer import _cfg, Mlp, Block
from timm.models.vision_transformer import trunc_normal_
from timm.models.registry import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn as nn
class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX


specification = {
    'LeViT_small': {
        'C': '384_512_512', 'D': 1, 'N': '2_2_2', 'X': '2_4_4', 'drop_path': 0.0,
        'weights': 'None'},
    'Trans4SOAR_base': {
        'C': '384_512_768', 'D': 32, 'N': '6_9_12', 'X': '4_4_4', 'drop_path': 0.1,
        'weights': 'None'},
}

__all__ = [specification.keys()]



@register_model
def Trans4SOAR_base(num_classes=512, distillation=True,
              pretrained=False, fuse=False, in_chans=3):
    return model_factory(**specification['Trans4SOAR_base'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse, in_chans=in_chans)
@register_model
def Trans4SOAR_small(num_classes=512, distillation=True,
              pretrained=False, fuse=False, in_chans=3):
    return model_factory(**specification['Trans4SOAR_small'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse, in_chans=in_chans)


FLOPS_COUNTER = 0


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

        global FLOPS_COUNTER
        output_points = ((resolution + 2 * pad - dilation *
                          (ks - 1) - 1) // stride + 1) ** 2
        FLOPS_COUNTER += a * b * output_points * (ks ** 2) // groups

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1), w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Linear_BN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1, resolution=-100000):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

        global FLOPS_COUNTER
        output_points = resolution ** 2
        FLOPS_COUNTER += a * b * output_points

    @torch.no_grad()
    def fuse(self):
        l, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape_as(x)
class Conv2d_BN_up(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()

        self.add_module('c', torch.nn.ConvTranspose2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

        global FLOPS_COUNTER
        output_points = ((resolution + 2 * pad - dilation *
                          (ks - 1) - 1) // stride + 1)**2
        FLOPS_COUNTER += a * b * output_points * (ks**2) // groups

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1), w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        l = torch.nn.Linear(a, b, bias=bias)
        trunc_normal_(l.weight, std=std)
        if bias:
            torch.nn.init.constant_(l.bias, 0)
        self.add_module('l', l)
        global FLOPS_COUNTER
        FLOPS_COUNTER += a * b

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def b16(n, activation, in_chans, resolution=224):
    return torch.nn.Sequential(
        Conv2d_BN(in_chans, n // 8, 3, 2, 1, resolution=resolution),
        activation(),
        Conv2d_BN(n // 8, n // 4, 3, 2, 1, resolution=resolution // 2),
        activation(),
        Conv2d_BN(n // 4, n // 2, 3, 2, 1, resolution=resolution // 4),
        activation(),
        Conv2d_BN(n // 2, n, 3, 2, 1, resolution=resolution // 8))


class Residual(torch.nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Linear_BN(dim, h, resolution=resolution)
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, dim, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

        global FLOPS_COUNTER
        # queries * keys
        FLOPS_COUNTER += num_heads * (resolution ** 4) * key_dim
        # softmax
        FLOPS_COUNTER += num_heads * (resolution ** 4)
        # attention * v
        FLOPS_COUNTER += num_heads * self.d * (resolution ** 4)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x, save_option=True):  # x (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, self.num_heads, -
        1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (
                (q @ k.transpose(-2, -1)) * self.scale
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        #if save_option == True:
        #    name = str(time.time()).replace('.', '-')
        #    torch.save(attn, '/home/kpeng/oneshot_metriclearning/transformer-sl-dml/save_attention_noise/'+name+'.pt')
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class Subsample(torch.nn.Module):
    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, self.resolution, self.resolution, C)[
            :, ::self.stride, ::self.stride].reshape(B, -1, C)
        return x


class AttentionSubsample(torch.nn.Module):
    def __init__(self, in_dim, out_dim, key_dim, num_heads=8,
                 attn_ratio=2,
                 activation=None,
                 stride=2,
                 resolution=14, resolution_=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_ ** 2
        h = self.dh + nh_kd
        self.kv = Linear_BN(in_dim, h, resolution=resolution)

        self.q = torch.nn.Sequential(
            Subsample(stride, resolution),
            Linear_BN(in_dim, nh_kd, resolution=resolution_))
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, out_dim, resolution=resolution_))

        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(itertools.product(
            range(resolution_), range(resolution_)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                    abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_, N))

        global FLOPS_COUNTER
        # queries * keys
        FLOPS_COUNTER += num_heads * \
                         (resolution ** 2) * (resolution_ ** 2) * key_dim
        # softmax
        FLOPS_COUNTER += num_heads * (resolution ** 2) * (resolution_ ** 2)
        # attention * v
        FLOPS_COUNTER += num_heads * \
                         (resolution ** 2) * (resolution_ ** 2) * self.d

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape
        k, v = self.kv(x).view(B, N, self.num_heads, -
        1).split([self.key_dim, self.d], dim=3)
        k = k.permute(0, 2, 1, 3)  # BHNC
        v = v.permute(0, 2, 1, 3)  # BHNC
        q = self.q(x).view(B, self.resolution_2, self.num_heads,
                           self.key_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + \
               (self.attention_biases[:, self.attention_bias_idxs]
                if self.training else self.ab)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        x = self.proj(x)
        return x


class MF(nn.Module):
    def __init__(self, dim, num_heads=64, qkv_bias=True, qk_scale=0.8, attn_drop=0., proj_drop=0.0, class_token=None, num_patches=384, drop_rate=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq_jb = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk_jb = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv_jb = nn.Linear(dim, dim, bias=qkv_bias)
        self.num_patches = num_patches
        self.wq_jv = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk_jv = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv_jv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.cls_token = class_token
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_tokens = 1 if class_token else 0
        self.cls_token_j = nn.Parameter(torch.zeros(1, 1, dim)) if self.num_tokens > 0 else None
        self.pos_embed_j = nn.Parameter(torch.randn(1, num_patches + self.num_tokens,dim) * .02)
        self.cls_token_v = nn.Parameter(torch.zeros(1, 1, dim)) if self.num_tokens > 0 else None
        self.pos_embed_v = nn.Parameter(torch.randn(1, num_patches + self.num_tokens,dim) * .02)
        self.cls_token_b = nn.Parameter(torch.zeros(1, 1, dim)) if self.num_tokens > 0 else None
        self.pos_embed_b = nn.Parameter(torch.randn(1, num_patches + self.num_tokens,dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.fuse_q = nn.Linear(2*num_patches, num_patches)
        self.fuse_k = nn.Linear(2*2*num_patches,2*num_patches)
        self.fuse_v = nn.Linear(2*2*num_patches,2*num_patches)
    def forward(self, joints, bones, velocity):
        joints = joints.permute(0,2,1)
        bones = bones.permute(0,2,1)
        velocity = velocity.permute(0,2,1)
        if self.cls_token is not None:
            joints = torch.cat((self.cls_token_j.expand(joints.shape[0], -1, -1), joints), dim=1)
            bones = torch.cat((self.cls_token_b.expand(joints.shape[0], -1, -1), bones), dim=1)
            velocity = torch.cat((self.cls_token_v.expand(velocity.shape[0], -1, -1), velocity), dim=1)
        joints = self.pos_drop(joints + self.pos_embed_j)
        bones = self.pos_drop(bones + self.pos_embed_b)
        velocity = self.pos_drop(velocity + self.pos_embed_v)
        B, N, C = joints.size()

        #freqs_h = pos_emb(torch.linspace(-1, 1, steps = 256), cache_key = 256)
        #freqs_w = pos_emb(torch.linspace(-1, 1, steps = 256), cache_key = 256)
        q_jb = self.wq_jb(joints).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BHN(C/H)
        k_jb = self.wk_jb(torch.cat([joints,bones], dim=1)).reshape(B, 2*N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BH2N(C/H)
        v_jb = self.wv_jb(torch.cat([joints,bones],dim=1)).reshape(B, 2*N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BH2N(C/H)

        q_jv = self.wq_jv(joints).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BHN(C/H)
        k_jv = self.wk_jv(torch.cat([joints,velocity], dim=1)).reshape(B, 2*N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BH2N(C/H)
        v_jv = self.wv_jv(torch.cat([joints,velocity],dim=1)).reshape(B, 2*N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BH2N(C/H)
        #print(torch.cat([q_jb, q_jv], dim=2).permute(0,1,3,2).size())
        #print(self.fuse_q)
        #q = self.fuse_q(torch.cat([q_jb, q_jv], dim=2).permute(0,1,3,2)).permute(0,1,3,2)
        #k = self.fuse_k(torch.cat([k_jb, k_jv], dim=2).permute(0,1,3,2)).permute(0,1,3,2)
        #v = self.fuse_v(torch.cat([v_jb, v_jv], dim=2).permute(0,1,3,2)).permute(0,1,3,2)
        q = (q_jb*torch.nn.functional.softmax(q_jv,dim=-1) + q_jv*torch.nn.functional.softmax(q_jb,dim=-1))/2
        k = (k_jb*torch.nn.functional.softmax(k_jv, dim=-1) + k_jv*torch.nn.functional.softmax(k_jb, dim=-1))/2
        v = (v_jb*torch.nn.functional.softmax(v_jv, dim=-1) + v_jv*torch.nn.functional.softmax(v_jb, dim=-1))/2
        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BHN2N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().view(B, -1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> BNC
        x = self.proj(x)
        x = self.proj_drop(x)
        return  x.permute(0,2,1)


class MAFM(nn.Module):

    def __init__(self, dim, num_heads=16, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer([256,384])
        self.attn = MF(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer([256,384])
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, joints, velocities, bones):

        x = (joints + velocities + bones)/3 + self.drop_path(self.attn(self.norm1(joints), self.norm1(velocities), self.norm1(bones)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x).permute(0,2,1))).permute(0,2,1)

        return x

class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)



class main_model(torch.nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=256,
                 patch_size=16,
                 in_chans=3,
                 num_classes=512,
                 embed_dim=[192],
                 key_dim=[64],
                 depth=[12],
                 num_heads=[3],
                 attn_ratio=[2],
                 mlp_ratio=[2],
                 hybrid_backbone=None,
                 down_ops=[],
                 attention_activation=Hardswish,
                 mlp_activation=Hardswish,
                 distillation=True,
                 drop_path=0):
        super().__init__()
        global FLOPS_COUNTER

        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.distillation = distillation
        act = Hardswish

        self.patch_embed = hybrid_backbone
        import copy
        self.patch_embed_joints = hybrid_backbone
        self.patch_embed_velocity = copy.deepcopy(hybrid_backbone)
        self.patch_embed_bones = copy.deepcopy(hybrid_backbone)
        self.mixed_fusion = MAFM(dim=256)
        self.blocks = []
        down_ops.append([''])
        resolution = img_size // patch_size
        self.intermediate_mark = []
        self.intermediate_mark_2 = []
        self.blocks_2 = []
        self.count = 0
        self.count_2 = 0
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            for _ in range(dpth):
                self.blocks.append(
                    Residual(Attention(
                        ed, kd, nh,
                        attn_ratio=ar,
                        activation=attention_activation,
                        resolution=resolution,
                    ), drop_path))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(ed, h, resolution=resolution),
                            mlp_activation(),
                            Linear_BN(h, ed, bn_weight_init=0,
                                      resolution=resolution),
                        ), drop_path))
                self.count += 1
                if mr > 0:
                    self.count += 1
            # self.intermediate_mark.append(self.count)

            if do[0] == 'Subsample':
                # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks.append(
                    AttentionSubsample(
                        *embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2],
                        attn_ratio=do[3],
                        activation=attention_activation,
                        stride=do[5],
                        resolution=resolution,
                        resolution_=resolution_))
                resolution = resolution_
                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(embed_dim[i + 1], h,
                                      resolution=resolution),
                            mlp_activation(),
                            Linear_BN(
                                h, embed_dim[i + 1], bn_weight_init=0, resolution=resolution),
                        ), drop_path))
                    self.count += 1
            self.count += 1
            self.intermediate_mark.append(self.count - 1)
        resolution = img_size // patch_size
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            for _ in range(dpth):
                self.blocks_2.append(
                    Residual(Attention(
                        ed, kd, nh,
                        attn_ratio=ar,
                        activation=attention_activation,
                        resolution=resolution,
                    ), drop_path))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks_2.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(ed, h, resolution=resolution),
                            mlp_activation(),
                            Linear_BN(h, ed, bn_weight_init=0,
                                      resolution=resolution),
                        ), drop_path))
                self.count_2 += 1
                if mr > 0:
                    self.count_2 += 1
            # self.intermediate_mark.append(self.count)

            if do[0] == 'Subsample':
                # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks_2.append(
                    AttentionSubsample(
                        *embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2],
                        attn_ratio=do[3],
                        activation=attention_activation,
                        stride=do[5],
                        resolution=resolution,
                        resolution_=resolution_))
                self.count_2 += 1
                resolution = resolution_
                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks_2.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(embed_dim[i + 1], h,
                                      resolution=resolution),
                            mlp_activation(),
                            Linear_BN(
                                h, embed_dim[i + 1], bn_weight_init=0, resolution=resolution),
                        ), drop_path))
                    self.count_2 += 1
                    emb = embed_dim[i+1]
                    #print(emb)
            self.count_2 += 1
            self.intermediate_mark_2.append(self.count_2 - 1)
            #print(self.count-1)
            self.blocks_2.append(torch.nn.Linear(emb, emb))
            self.count_2 += 1
        #print(len(self.blocks))
        #print(len(self.blocks_2))
        #sys.exit()
        self.blocks = torch.nn.Sequential(*self.blocks)
        self.blocks_2 = torch.nn.Sequential(*self.blocks_2)
        # print(self.intermediate_mark)
        #self.fuse = MLP([512, 21])
        #self.merge = MLP([1024, 512, 512])

        #self.fuse_2 = MLP([384, 21])
        #self.merge_2 = MLP([1024, 512, 512])
        #self.causal_1 = causal_aggregate(64)
        #self.causal_2 = causal_aggregate(16)
        self.reconstruction_loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

        # Classifier head
        self.head = BN_Linear(
            embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.head_2 = BN_Linear(
            embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        if distillation:
            self.head_dist = BN_Linear(
                embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.head_inter = BN_Linear(
            512, num_classes) if num_classes > 0 else torch.nn.Identity()
        if distillation:
            self.head_dist = BN_Linear(
                512, num_classes) if num_classes > 0 else torch.nn.Identity()
        #self.infonce_loss = info_nce.InfoNCE()
        #self.cosin_loss = torch.nn.CosineEmbeddingLoss(margin=0.1, size_average=None, reduce=None, reduction='mean')
        self.FLOPS = FLOPS_COUNTER
        FLOPS_COUNTER = 0
        self.cosine_embedding_loss = torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
        
        self.f_feature = nn.Linear(512, 768)
        self.f_prototypes = nn.Linear(512, 768)
        self.softmax = nn.Softmax()
        self.f_a = nn.Linear(768*2, 768)
        self.f_r = nn.Linear(768, 512)
        self.relu = nn.ReLU()
        self.fc_fuse = nn.Linear(384*3, 384)

        '''
        
        self.f_feature = nn.Linear(256, 512)
        self.f_prototypes = nn.Linear(256, 512)
        self.softmax = nn.Softmax()
        self.f_a = nn.Linear(512*2, 512)
        self.f_r = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        '''
    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}


    def get_augmented_feature(self, features, labels, prototypes, epoch):
        if epoch > 20:
            curbatch_proto = torch.nn.functional.softmax(torch.stack([prototypes[i,:] for i in labels], dim=0),dim=-1)*features #torch.stack([prototypes[i,:] for i in labels], dim=0)*0.5 + 0.5*features
            #curbatch_proto = torch.stack([prototypes[i,:] for i in labels], dim=0)
        else:
            curbatch_proto = features
        lift_proto = self.f_prototypes(curbatch_proto)
        lift_feature = self.f_feature(features)
        attention = self.softmax(lift_feature*lift_proto)
        out_phi_a = self.relu(self.f_a(torch.cat([lift_proto*attention,lift_feature], dim=-1)))
        return self.relu(features+self.f_r(out_phi_a))

    def forward(self, x, prototypes=None, epoch=None, labels=None):

        ori_joints= x[:,0,...]
        ori_bone = x[:,1,...]
        ori_velocity = x[:,2,...]

        joints = ori_joints
        bone = ori_bone
        velocity = ori_velocity
        j = self.patch_embed_joints(joints).flatten(2).transpose(1, 2)
        b = self.patch_embed_bones(bone).flatten(2).transpose(1, 2)
        v = self.patch_embed_velocity(velocity).flatten(2).transpose(1, 2)
        x = self.mixed_fusion(j,b,v)
        if (prototypes !=None) and (epoch>1):
            x_2 = x
            count = 0
            c = 0
            for block in self.blocks:
                x = block(x)
                c+=1
                if c == (self.intermediate_mark[0]):
                    rep = x
                    #print(rep.size())
            x = torch.mean(x,dim=1,keepdim=False)
            final = self.head(x)
            x=final
            for block_2 in self.blocks_2:
                count += 1
                if count == self.intermediate_mark_2[0]:
                    aug_x_2 = self.get_augmented_feature(x_2, labels, prototypes, epoch)
                    x_2 = block_2(aug_x_2)
                else:
                    x_2 = block_2(x_2)
            x_2 = torch.mean(x_2, dim=1, keepdim=False)
            x_2 = self.head_2(x_2)
            target = torch.ones(x_2.size()[0]).cuda()
            loss = torch.cosine_embedding_loss(x, x_2,target)
            if epoch == 0:
                loss = torch.zeros(x.size()[0]).cuda()
        else:

            if epoch !=None:
                x = x
                c = 0
                for idx, block in enumerate(self.blocks):
                    x = block(x)
                    c += 1
                    if c == self.intermediate_mark[0]:
                        rep = x
                x = torch.mean(x, dim=1, keepdim=False)
                x = self.head(x)
                loss = torch.zeros(x.size()[0]).cuda()
            else:
                x = x
                c = 0
                for idx, block in enumerate(self.blocks):
                    x = block(x)
                    c += 1
                    if c == self.intermediate_mark[0]:
                        rep = x
                x = torch.mean(x, dim=1, keepdim=False)
                x = self.head(x)
                loss = torch.zeros(x.size()[0]).cuda()
        return x, loss, rep

def model_factory(C, D, X, N, drop_path, weights,
                  num_classes, distillation, pretrained, fuse, in_chans=3):
    embed_dim = [int(x) for x in C.split('_')]
    num_heads = [int(x) for x in N.split('_')]
    depth = [int(x) for x in X.split('_')]
    act = Hardswish
    model = main_model(
        patch_size=16,
        in_chans=in_chans,
        embed_dim=embed_dim,
        num_heads=num_heads,
        key_dim=[D] * 3,
        depth=depth,
        attn_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        down_ops=[
            # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', D, embed_dim[0] // D, 4, 2, 2],
            ['Subsample', D, embed_dim[1] // D, 4, 2, 2],
        ],
        attention_activation=act,
        mlp_activation=act,
        hybrid_backbone=b16(embed_dim[0], activation=act, in_chans=in_chans),
        num_classes=num_classes,
        drop_path=drop_path,
        distillation=distillation
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            weights, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    if fuse:
        utils.replace_batchnorm(model)

    return model

if __name__ == '__main__':
    for name in specification:
        net = globals()[name](fuse=True, pretrained=True)
        net.eval()
        net(torch.randn(4, 3, 224, 224))
        print(name,
              net.FLOPS, 'FLOPs',
              sum(p.numel() for p in net.parameters() if p.requires_grad), 'parameters')
