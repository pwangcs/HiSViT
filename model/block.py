import torch
from torch import nn 
import einops
from torch.nn import functional as F
import math
import torch.utils.checkpoint as cp

class STConv3d(nn.Module):
    def __init__(self, dim, groups=1, bias=True):
        super().__init__()
        self.sw_conv = nn.Sequential(
            nn.Conv3d(dim, dim, (1, 3, 3), 1, (0, 1, 1), groups=groups, bias=bias),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, dim, (1, 3, 3), 1, (0, 1, 1), groups=groups, bias=bias),
            nn.LeakyReLU(inplace=True),
        )
        self.tw_conv = nn.Conv3d(dim, dim, (3, 1, 1), 1, (1, 0, 0), groups=groups, bias=bias)

    def forward(self, x):     
        x1 = self.sw_conv(x)
        x2 = self.tw_conv(x)
        y = x1 + x2
        return y

class GSM_FFN(nn.Module):
    def __init__(self, dim, exp_ratio=2, act_layer=nn.GELU, bias=True):
        super().__init__()
        self.proj_in = nn.Conv3d(dim, int(dim*exp_ratio), kernel_size=1, bias=bias)
        self.dwconv = STConv3d(int(dim*exp_ratio)//2)
        self.proj_out = nn.Conv3d(int(dim*exp_ratio)//2, dim, kernel_size=1, bias=bias)
        self.act = act_layer()
    def forward(self, x):
        x = self.act(self.proj_in(x))
        x1, x2 = x.chunk(2,dim=1)
        x = self.dwconv(x1) * torch.sigmoid(x2)
        x = self.proj_out(x)
        return x


class CSS_MSA(nn.Module):
    """ Cross-Scale Separable Multi-head Self-Attention.
    Args:
        dim (int): Number of input channels.
        idx (int): The indentix of window. (0/1)
        split_size (tuple(int)): Height and Width of spatial window.
        num_heads (int): Number of attention heads. Default: 6
        attn_drop (float): Dropout ratio of attention weight. Default: 0.0
    """ 
    def __init__(self, dim, idx, num_heads=6, frames=8, split_size=[4,8],   pool_kernel=1, attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.frames = frames
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.pool_kernel = pool_kernel
        logit_scale = torch.log(10 * torch.ones((num_heads, 1, 1)))
        self.logit_scale_tp = nn.Parameter(logit_scale, requires_grad=True)
        self.logit_scale_sp = nn.Parameter(logit_scale, requires_grad=True)
        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        self.attn_drop1 = nn.Dropout(attn_drop)
        self.attn_drop2 = nn.Dropout(attn_drop)

    def vid2win(self, x, u, v):
        x = einops.rearrange(x,'b (d c) t (u h) (v w)-> (b t u v) d (h w) c', d=self.num_heads, u=u, v=v)
        return x
    
    def vid2seq(self, x):
        x = einops.rearrange(x, 'b (d c) t h w -> (b h w) d t c', d=self.num_heads)
        return x
        
    def forward(self, q, k, v, k_=None, mask=None):
        """
        Input: q, k, v: (3, B, C, T, H, W), 
                q: (B, C, T, H, W)
                mask: (num_Wins, N, N)
        Output: x (B, T, H, W, C)
        """
        B, C, T, H, W = v.shape
        
        num_H = H // self.H_sp
        num_W = W // self.W_sp

        # video to windows
        q_sp = self.vid2win(q, num_H, num_W)
        if k_ is not None:
            k_sp = self.vid2win(k_, num_H, num_W)
        else:
            k_sp = self.vid2win(k, num_H, num_W)
        v = self.vid2win(v, num_H, num_W)
        q_tp = self.vid2seq(q)
        k_tp = self.vid2seq(k)

        # sptatial attention
        attn_sp = F.normalize(q_sp, dim=-1) @ F.normalize(k_sp, dim=-1).transpose(-2, -1)
        attn_sp = attn_sp * torch.clamp(self.logit_scale_sp, max=math.log(1.0 / 0.01)).exp()
        # use mask for shift window
        if mask is not None:
            attn_sp = einops.rearrange(attn_sp, '(b t u v) d hw1 hw2-> (b t) d (u v) hw1 hw2', b=B, u=num_H, v=num_W)
            attn_sp = attn_sp.masked_fill_(mask, float("-inf"))
            attn_sp = einops.rearrange(attn_sp, 'bt d uv hw1 hw2-> (bt uv) d hw1 hw2')

        # temporal attention
        attn_tp = F.normalize(q_tp, dim=-1) @ F.normalize(k_tp, dim=-1).transpose(-2, -1)
        attn_tp = attn_tp * torch.clamp(self.logit_scale_tp, max=math.log(1.0 / 0.01)).exp()

        # Softmax
        attn_sp = F.softmax(attn_sp, dim=-1, dtype=attn_sp.dtype)       
        attn_tp = F.softmax(attn_tp, dim=-1, dtype=attn_tp.dtype)
        attn_sp = self.attn_drop1(attn_sp)
        attn_tp = self.attn_drop2(attn_tp)
        
        x = attn_sp @ v
        x = einops.rearrange(x, '(b t u v) d (h w) c-> (b u h v w) d t c', b=B, t=T, u=num_H, v=num_W, h=self.H_sp*self.pool_kernel, w=self.W_sp*self.pool_kernel)
        x = attn_tp @ x
        x = einops.rearrange(x, '(b u h v w) d t c-> b (d c) t (u h) (v w)', b=B, u=num_H, v=num_W, h=self.H_sp*self.pool_kernel, w=self.W_sp*self.pool_kernel)

        return x


class CSS_MSA_block(nn.Module):
    # The implementation of rectangle-window CSS-MSA, based on https://github.com/Zhengchen1999/CAT 
    """ 
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 6
        split_size (tuple(int)): Height and Width of spatial window.
        shift_size (tuple(int)): Shift size for spatial window.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Attention dropout rate. Default: 0.0
        shift_flag (bool): Spatial shift or Not
    """
    def __init__(self, dim, num_heads, reso, frames, split_size=[4,16],
                 pool_kernel=1, qkv_bias=True,
                 proj_drop=0., attn_drop=0., shift_flag=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.q_split_size = [int(split_size[0]* pool_kernel), int(split_size[1]* pool_kernel)]
        self.shift_size = [split_size[0]//2, split_size[1]//2]
        self.q_shift_size = [self.q_split_size[0]//2, self.q_split_size[1]//2]

        self.shift_flag = shift_flag
        self.patches_resolution = reso
        self.pool_kernel = pool_kernel

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)

        assert 0 <= self.shift_size[0] < self.split_size[0], "shift_size must in 0-split_size0"
        assert 0 <= self.shift_size[1] < self.split_size[1], "shift_size must in 0-split_size1"

        self.branch_num = 2

        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attns = nn.ModuleList([
                CSS_MSA(dim=self.dim//2,
                        idx=i,
                        num_heads=num_heads//2,
                        frames=frames,
                        split_size=split_size,
                        pool_kernel=pool_kernel,
                        attn_drop=attn_drop)
                for i in range(self.branch_num)])

        if self.shift_flag:
            attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)
            self.register_buffer("attn_mask_0", attn_mask[0]) 
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            attn_mask = None
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

    def calculate_mask(self, H, W):
        num_H_0, num_W_0 = H // self.q_split_size[0], W // self.q_split_size[1]
        num_H_1, num_W_1 = H // self.q_split_size[1], W // self.q_split_size[0]

        attn_mask_0 = torch.zeros(num_H_0, num_W_0, self.q_split_size[0], self.q_split_size[1], self.split_size[0], self.split_size[1], dtype=torch.bool)
        attn_mask_1 = torch.zeros(num_H_1, num_W_1, self.q_split_size[1], self.q_split_size[0], self.split_size[1], self.split_size[0], dtype=torch.bool)

        attn_mask_0[-1, :, :self.q_shift_size[0], :, self.shift_size[0]:, :] = True
        attn_mask_0[-1, :, self.q_shift_size[0]:, :, :self.shift_size[0], :] = True
        attn_mask_0[:, -1, :, :self.q_shift_size[1], :, self.shift_size[1]:] = True
        attn_mask_0[:, -1, :, self.q_shift_size[1]:, :, :self.shift_size[1]] = True

        attn_mask_1[-1, :, :self.q_shift_size[1], :, self.shift_size[1]:, :] = True
        attn_mask_1[-1, :, self.q_shift_size[1]:, :, :self.shift_size[1], :] = True
        attn_mask_1[:, -1, :, :self.q_shift_size[0], :, self.shift_size[0]:] = True
        attn_mask_1[:, -1, :, self.q_shift_size[0]:, :, :self.shift_size[0]] = True
        
        attn_mask_0 = einops.rearrange(attn_mask_0, 'w1 w2 p1 p2 p3 p4 ->1 1 (w1 w2) (p1 p2) (p3 p4)')
        attn_mask_1 = einops.rearrange(attn_mask_1, 'w1 w2 p1 p2 p3 p4 ->1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask_0, attn_mask_1


    def forward(self, x, T, H, W):
        B, L, C = x.shape
        assert L == T * H * W, "flattened image tokens have the wrong size"
        qkv = self.qkv(x)
        qkv = einops.rearrange(qkv, 'b (t h w) (n c)-> n b c t h w', t=T, h=H, w=W, n=3)
        q,k,v = qkv[0], qkv[1], qkv[2]

        k_ = None
        if self.pool_kernel > 1:
            k_ = einops.rearrange(k,'b c t h w-> (b t) c h w')
            k_ = F.avg_pool2d(k_, kernel_size=self.pool_kernel)
            k_ = einops.rearrange(k_,'(b t) c h w-> b c t h w', b=B,t=T)
            v = einops.rearrange(v,'b c t h w-> (b t) c h w')
            v = F.avg_pool2d(v, kernel_size=self.pool_kernel)
            v = einops.rearrange(v,'(b t) c h w-> b c t h w', b=B,t=T)
            H_, W_ = v.shape[-2], v.shape[-1]
        else:
            H_, W_ = H, W
        ## image padding
        max_split_size = max(self.split_size[0], self.split_size[1])
        if H_ % max_split_size != 0 or W_ % max_split_size != 0:
            pad_l = pad_t = 0
            pad_r = (max_split_size - W_ % max_split_size) % max_split_size
            pad_b = (max_split_size - H_ % max_split_size) % max_split_size
            q = F.pad(q, (pad_l, pad_r, pad_t*self.pool_kernel, pad_b*self.pool_kernel))
            k = F.pad(k, (pad_l, pad_r, pad_t*self.pool_kernel, pad_b*self.pool_kernel))
            v = F.pad(v, (pad_l, pad_r, pad_t, pad_b))
            if k_ is not None:
                k_ = F.pad(k_, (pad_l, pad_r, pad_t, pad_b))
            H_pad = pad_b + H_
            W_pad = pad_r + W_
        else:
            H_pad = H_
            W_pad = W_

        ## window-0 and window-1 on split channels [C/2, C/2]
        if self.shift_flag:
            q_0 = torch.roll(q[:,:C//2,:,:,:], shifts=(-self.q_shift_size[0], -self.q_shift_size[1]), dims=(-2, -1))
            q_1 = torch.roll(q[:,C//2:,:,:,:], shifts=(-self.q_shift_size[1], -self.q_shift_size[0]), dims=(-2, -1))
            k_0 = torch.roll(k[:,:C//2,:,:,:], shifts=(-self.q_shift_size[0], -self.q_shift_size[1]), dims=(-2, -1))
            k_1 = torch.roll(k[:,C//2:,:,:,:], shifts=(-self.q_shift_size[1], -self.q_shift_size[0]), dims=(-2, -1))
            v_0 = torch.roll(v[:,:C//2,:,:,:], shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(-2, -1))
            v_1 = torch.roll(v[:,C//2:,:,:,:], shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(-2, -1))
            if self.pool_kernel > 1:
                k__0 = torch.roll(k_[:,:C//2,:,:,:], shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(-2, -1))
                k__1 = torch.roll(k_[:,C//2:,:,:,:], shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(-2, -1))
            else:
                k__0 = None
                k__1 = None
            if self.patches_resolution != (H_pad* self.pool_kernel) or self.patches_resolution != (W_pad* self.pool_kernel):
                
                print('Mask is generated again')
                mask_tmp = self.calculate_mask(H_pad* self.pool_kernel, W_pad* self.pool_kernel)
                x1_shift = self.attns[0](q_0, k_0, v_0, k__0, mask=mask_tmp[0].to(x.device))
                x2_shift = self.attns[1](q_1, k_1, v_1, k__1, mask=mask_tmp[1].to(x.device))
            else:
                x1_shift = self.attns[0](q_0, k_0, v_0, k__0, mask=self.attn_mask_0)
                x2_shift = self.attns[1](q_1, k_1, v_1, k__1, mask=self.attn_mask_1)
            x1 = torch.roll(x1_shift, shifts=(self.q_shift_size[0], self.q_shift_size[1]), dims=(-2, -1))[:, :, :, :H, :W]
            x2 = torch.roll(x2_shift, shifts=(self.q_shift_size[1], self.q_shift_size[0]), dims=(-2, -1))[:, :, :, :H, :W]
            # attention output
            attened_x = torch.cat([x1,x2], dim=1)
        else:
            if self.pool_kernel > 1:
                x1 = self.attns[0](q[:,:C//2,:,:,:], k[:,:C//2,:,:,:], v[:,:C//2,:,:,:], k_[:,:C//2,:,:,:])[:, :, :, :H, :W]
                x2 = self.attns[1](q[:,C//2:,:,:,:], k[:,C//2:,:,:,:], v[:,C//2:,:,:,:], k_[:,C//2:,:,:,:])[:, :, :, :H, :W]
            else:
                x1 = self.attns[0](q[:,:C//2,:,:,:], k[:,:C//2,:,:,:], v[:,:C//2,:,:,:])[:, :, :, :H, :W]
                x2 = self.attns[1](q[:,C//2:,:,:,:], k[:,C//2:,:,:,:], v[:,C//2:,:,:,:])[:, :, :, :H, :W]                         
            # attention output
            attened_x = torch.cat([x1,x2], dim=1)

        x = attened_x.reshape(B,C,-1).transpose(1,2).contiguous()

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ViTblock(nn.Module):
    def __init__(self, dim, num_heads, reso=128, frames=8, split_size=[4,16], pool_kernel=1, qkv_bias=True, exp_ratio=2, proj_drop=0., attn_drop=0., shift_flag=False):
        super().__init__()
        
        self.attn = CSS_MSA_block(dim=dim,
                                  num_heads=num_heads,
                                  reso=reso,
                                  frames=frames,
                                  split_size=split_size,
                                  pool_kernel=pool_kernel,
                                  qkv_bias=qkv_bias,
                                  proj_drop=proj_drop,
                                  attn_drop=attn_drop,
                                  shift_flag=shift_flag)

        self.ffn = GSM_FFN(dim=dim,exp_ratio=exp_ratio)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        T, H, W = x.shape[-3::]

        x = einops.rearrange(x, 'b c t h w-> b (t h w) c').contiguous()
        x = x + self.attn(self.norm1(x), T, H, W)
        shortcut = einops.rearrange(x, 'b (t h w) c-> b c t h w', t=T, h=H, w=W).contiguous()
        x = einops.rearrange(self.norm2(x), 'b (t h w) c-> b c t h w', t=T, h=H, w=W).contiguous()
        x = shortcut + self.ffn(x)

        return x     