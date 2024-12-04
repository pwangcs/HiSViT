import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from distutils.version import LooseVersion
import numpy as np
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as cp

def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)

def window_partition(x, window_size):
    """ Partition the input into windows. Attention will be conducted within the windows.

    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)

    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """ Reverse windows back to the original input. Attention was conducted within the windows.

    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)

    return x


def get_window_size(x_size, window_size, shift_size=None):
    """ Get the window size and the shift size """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    """ Compute attnetion mask for input of size (D, H, W). @lru_cache caches each stage results. """

    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class Mlp(nn.Module):
    """ Multilayer perceptron.

    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    # def forward(self, x):
    #     return self.fc2(self.act(self.fc1(x)))
    
    def mlp(self, x):
        return self.fc2(self.act(self.fc1(x)))
    def forward(self, x):
        return cp.checkpoint(self.mlp, x, preserve_rng_state=False, use_reentrant=False)


class WindowAttention(nn.Module):
    """ Window based multi-head self attention.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH
        self.register_buffer("relative_position_index", self.get_position_index(window_size))
        self.qkv_self = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """

        # self attention
        B_, N, C = x.shape
        qkv = self.qkv_self(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C
        x_out = self.attention(q, k, v, mask, (B_, N, C))

        # projection
        x = self.proj(x_out)

        return x

    def attention(self, q, k, v, mask, x_shape):
        B_, N, C = x_shape
        attn = (q * self.scale) @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)  # Wd*Wh*Ww, Wd*Wh*Ww,nH
        attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask[:, :N, :N].unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, -1, dtype=q.dtype)  # Don't use attn.dtype after addition!
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        return x

    def get_position_index(self, window_size):
        ''' Get pair-wise relative position index for each token inside the window. '''

        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww

        return relative_position_index


class STL(nn.Module):
    """ Swin Transformer Layer (STL).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for mutual and self attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=(2, 8, 8),
                 shift_size=(0, 0, 0),
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint_attn = use_checkpoint_attn
        self.use_checkpoint_ffn = use_checkpoint_ffn

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1), mode='constant')

        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C

        # attention / shifted attention
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C

        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]

        return x

    def forward_part2(self, x):
        return self.mlp(self.norm2(x))

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        # attention
        if self.use_checkpoint_attn:
            x = x + checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = x + self.forward_part1(x, mask_matrix)

        # feed-forward
        if self.use_checkpoint_ffn:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class STG(nn.Module):
    """ Swin Transformer Group (STG).

    Args:
        dim (int): Number of feature channels
        input_resolution (tuple[int]): Input resolution.
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (6,8,8).
        shift_size (tuple[int]): Shift size for mutual and self attention. Default: None.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=[2, 8, 8],
                 shift_size=None,
                 mlp_ratio=2.,
                 qkv_bias=False,
                 qk_scale=None,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = list(i // 2 for i in window_size) if shift_size is None else shift_size

        # build blocks
        self.blocks = nn.ModuleList([
            STL(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0, 0, 0] if i % 2 == 0 else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                use_checkpoint_attn=use_checkpoint_attn,
                use_checkpoint_ffn=use_checkpoint_ffn
            )
            for i in range(depth)])

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for attention
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = x.view(B, D, H, W, -1)
        x = rearrange(x, 'b d h w c -> b c d h w')

        return x


class RSTB(nn.Module):
    """ Residual Swin Transformer Block (RSTB), https://github.com/JingyunLiang/SwinIR

    Args:
        kwargs: Args for RSTB.
    """

    def __init__(self, **kwargs):
        super(RSTB, self).__init__()
        self.input_resolution = kwargs['input_resolution']

        self.residual_group = STG(**kwargs)
        self.linear = nn.Linear(kwargs['dim'], kwargs['dim'])

    def forward(self, x):
        return x + self.linear(self.residual_group(x).transpose(1, 4)).transpose(1, 4)


class RSTBWithInputConv(nn.Module):
    """RSTB with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        kernel_size (int): Size of kernel of the first conv.
        stride (int): Stride of the first conv.
        group (int): Group of the first conv.
        num_blocks (int): Number of residual blocks. Default: 2.
         **kwarg: Args for RSTB.
    """

    def __init__(self, in_channels=3, kernel_size=(1, 3, 3), stride=1, groups=1, num_blocks=2, **kwargs):
        super().__init__()

        main = []
        main += [Rearrange('n d c h w -> n c d h w'),
                 nn.Conv3d(in_channels,
                           kwargs['dim'],
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2),
                           groups=groups),
                 Rearrange('n c d h w -> n d h w c'),
                 nn.LayerNorm(kwargs['dim']),
                 Rearrange('n d h w c -> n c d h w')]

        # RSTB blocks
        kwargs['use_checkpoint_attn'] = kwargs.pop('use_checkpoint_attn')[0]
        kwargs['use_checkpoint_ffn'] = kwargs.pop('use_checkpoint_ffn')[0]
        main.append(make_layer(RSTB, num_blocks, **kwargs))

        main += [Rearrange('n c d h w -> n d h w c'),
                 nn.LayerNorm(kwargs['dim']),
                 Rearrange('n d h w c -> n d c h w')]

        self.main = nn.Sequential(*main)

    def forward(self, x):
        """
        Forward function for RSTBWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, t, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, t, out_channels, h, w)
        """
        return self.main(x)