# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import sys
import pickle
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=64, qkv_bias=True, qk_scale=0.8, attn_drop=0., proj_drop=0.0, class_token=None, num_patches=96, drop_rate=0.0):
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
        self.pos_embed_j = nn.Parameter(torch.randn(1, num_patches + self.num_tokens, dim) * .02)
        self.cls_token_v = nn.Parameter(torch.zeros(1, 1, dim)) if self.num_tokens > 0 else None
        self.pos_embed_v = nn.Parameter(torch.randn(1, num_patches + self.num_tokens, dim) * .02)
        self.cls_token_b = nn.Parameter(torch.zeros(1, 1, dim)) if self.num_tokens > 0 else None
        self.pos_embed_b = nn.Parameter(torch.randn(1, num_patches + self.num_tokens, dim) * .02)
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
        #print(joints.size())
        #print(self.pos_embed_j.size())
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


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads=16, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer([3136,96])
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer([3136,96])
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, joints, velocities, bones):

        x = (joints + velocities + bones)/3 + self.drop_path(self.attn(self.norm1(joints), self.norm1(velocities), self.norm1(bones)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x).permute(0,2,1)).permute(0,2,1))
        return x

class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        #with open('/home/kpeng/oneshot_metriclearning/transformer-sl-dml/word_embedding_ntu.pkl', 'rb') as f:
        #    self.world_embedding = pickle.load(f)

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.cross_attention = CrossAttentionBlock(dim=3136)

        # split image into non-overlapping patches
        self.patch_embed_joints = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.patch_embed_bone = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.patch_embed_velocity = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed_joints.num_patches
        patches_resolution = self.patch_embed_joints.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.layers_2 = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        for i_layer in range(self.num_layers):
            print(i_layer)
            #if i_layer == 0 or i_layer == 1:
            #    pass
            layer_2 = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers_2.append(layer_2)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_2 = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()


        self.apply(self._init_weights)
        self.cosine_embedding_loss = torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
        self.f_feature = nn.Linear(384, 256)
        self.f_prototypes = nn.Linear(384, 256)
        self.softmax = nn.Softmax()
        self.f_a = nn.Linear(256*2, 256)
        self.f_r = nn.Linear(256, 384)
        self.relu = nn.ReLU()
        #self.f_feature_2 = nn.Linear(512, 256)
        #self.f_prototypes_2 = nn.Linear(512, 256)
        #self.softmax_2 = nn.Softmax()
        #self.f_a_2 = nn.Linear(256*2, 256)
        #self.f_r_2 = nn.Linear(256, 512)
        #self.relu_2 = nn.ReLU()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        if epoch != None:
            ori_joints= torch.cat([x[:,0,...],x[:,3,...]], dim=0)
            ori_bone = torch.cat([x[:,1,...], x[:,4,...]], dim=0)
            ori_velocity = torch.cat([x[:,2,...], x[:,5,...]], dim=0)
        else:
            ori_joints = x[:,0,...]
            ori_bone = x[:,1,...]
            ori_velocity = x[:,2,...]
        joints = ori_joints
        bone = ori_bone
        velocity = ori_velocity
        #print(self.patch_embed_joints(joints).size())
        j = self.patch_embed_joints(joints)
        b = self.patch_embed_bone(bone)
        v = self.patch_embed_velocity(velocity)
        #print(j.size())
        x = self.cross_attention(j,b,v)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x
    def get_augmented_feature(self, features, labels, prototypes, epoch):
        if epoch > 0:
            curbatch_proto = torch.nn.functional.softmax(torch.stack([prototypes[i,:] for i in labels], dim=0), dim=-1)*features
        else:
            curbatch_proto = features
        lift_proto = self.f_prototypes(curbatch_proto)
        lift_feature = self.f_feature(features)
        attention = self.softmax(lift_feature*lift_proto)
        out_phi_a = self.relu(self.f_a(torch.cat([lift_proto*attention,lift_feature], dim=-1)))
        return self.relu(features+self.f_r(out_phi_a))

    def get_augmented_feature_with_semantic(self, features, labels, prototypes):
        curbatch_proto = torch.stack([prototypes[i,:] for i in labels], dim=0)
        curbatch_proto_semantic = torch.stack([self.world_embedding[i] for i in labels], dim = 0)
        lift_proto = self.f_prototypes(curbatch_proto)
        lift_feature = self.f_feature(features)
        attention = self.softmax(lift_feature*lift_proto)
        out_phi_a = self.relu(self.f_a(torch.cat([lift_proto*attention,lift_feature], dim=-1)))
        features = self.relu(features+self.f_r(out_phi_a))
        lift_proto = self.f_prototypes_2(curbatch_proto_semantic).squeeze().unsqueeze(1).repeat(1,36,1)

        lift_feature = self.f_feature_2(features)
        attention = self.softmax_2(lift_feature*lift_proto)
        out_phi_a = self.relu_2(self.f_a_2(torch.cat([lift_proto*attention,lift_feature], dim=-1)))
        features = self.relu(features + self.f_r_2(out_phi_a))
        return features


    def forward(self, x, prototypes=None, epoch=None, labels=None):

        if epoch != None:
            labels = torch.cat([labels,labels],dim=0)
            ori_joints = x[:,0,...]
            ori_bone = x[:,1,...]
            ori_velocity = x[:,2,...]
            ori_joints= torch.cat([x[:,0,...],x[:,3,...]], dim=0)
            ori_bone = torch.cat([x[:,1,...], x[:,4,...]], dim=0)
            ori_velocity = torch.cat([x[:,2,...], x[:,5,...]], dim=0)
        else:
            ori_joints = x[:,0,...]
            ori_bone = x[:,1,...]
            ori_velocity = x[:,2,...]
        joints = ori_joints
        bone = ori_bone
        velocity = ori_velocity
        #print(self.patch_embed_joints(joints).size())
        j = self.patch_embed_joints(joints)
        b = self.patch_embed_bone(bone)
        v = self.patch_embed_velocity(velocity)
        #print(j.size())
        #x  = j
        x = self.cross_attention(j,b,v)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        count = 0
        coun = 0
        x_ori = x
        #print(prototypes)
        #print(epoch)
        if prototypes == None:
            standard = prototypes == None
        else:
            standard = 0
        
        if epoch == None:
            epoch = 1
        else:
            epoch += 0
        
        if  (epoch ==1):
            for layer in self.layers:
                x = layer(x)
                #print(x.size())
                count+=1
                if count == 2:
                    rep = x
                    #print(rep.size())
        
        #print(x.size())
        #sys.exit()
        #print(epoch)
        if (~standard) and (epoch>1):
            for layer in self.layers:
                #print(x.size())
                x = layer(x)
                coun+=1
                if coun == 2:
                    rep = x
                #print(x.size())
            x_2 = x_ori
            for layer_2 in self.layers_2:
                #print(x_ori.size())
                x_2 = layer_2(x_2)
                count+=1
                if count == 2:
                    #print(x_2.size())
                    aug_x_2 = self.get_augmented_feature(x_2, labels, prototypes, epoch)
                    x_2 = aug_x_2
                    #print(x.size())
                    #sys.exit()
            x_2 = self.norm(x_2)
            x_2 = self.avgpool(x_2.transpose(1,2))
            x_2 = torch.flatten(x_2,1)
            #x_2 = torch.mean(x_2, dim=1, keepdim=False)
            x_2 = self.head_2(x_2)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        if (~standard) and (epoch > 1):
            target = torch.ones(x_2.size()[0]).cuda()
            loss = torch.cosine_embedding_loss(x, x_2, target)
        else:
            loss = torch.zeros(x.size()[0]).cuda()
        return x, loss, rep

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
def build_model():
    if 1:
        model = SwinTransformer(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=512,
                                embed_dim=96,
                                depths=[2,2,6,2],
                                num_heads=[3,6,12,24],
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0,
                                drop_path_rate=0.2,
                                ape=False,
                                patch_norm=nn.LayerNorm,
                                use_checkpoint=False)

    return model
