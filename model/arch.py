from torch import nn 
import torch
from einops.layers.torch import Rearrange
from model.block import ViTblock
from  model.rstb import RSTBWithInputConv


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=True):
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv3d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.conv_du(self.avg_pool(x))

class GroupBlock(nn.Module):
    def __init__(self, dim=256, dim_groups=[64,64,128], reso=128, frames=8, dim_head=16, idx=0):
        super().__init__()
        self.dim = dim
        self.dim_groups = dim_groups
        assert dim == sum(dim_groups), "dim doesn't match."
        self.conv_list = nn.ModuleList()
        self.vit_list = nn.ModuleList()
        self.vit_list.append(ViTblock(dim=dim_groups[2],
                                num_heads=dim_groups[2]//dim_head,
                                reso=reso, frames=frames,
                                split_size=[4,32], pool_kernel=4,
                                shift_flag=bool(idx % 2)))
        self.vit_list.append(ViTblock(dim=dim_groups[1],
                                num_heads=dim_groups[1]//dim_head,
                                reso=reso, frames=frames,
                                split_size=[4,16], pool_kernel=2,
                                shift_flag=bool(idx % 2)))
        self.vit_list.append(ViTblock(dim=dim_groups[0],
                                num_heads=dim_groups[0]//dim_head,
                                reso=reso, frames=frames,
                                split_size=[4,16], pool_kernel=1,
                                shift_flag=bool(idx % 2)))
        self.conv_list.append(nn.Sequential(
                        nn.Conv3d(dim_groups[2]+dim_groups[1],dim_groups[1],1),
                        nn.LeakyReLU(inplace=True)
                    ))
        self.conv_list.append(nn.Sequential(
                        nn.Conv3d(dim_groups[2]+dim_groups[1]+dim_groups[0],dim_groups[0],1),
                        nn.LeakyReLU(inplace=True)
                    ))
        self.last_conv = nn.Conv3d(self.dim, self.dim,1)
        self.channel_attn = CALayer(self.dim)

    def forward(self, x):
        assert x.shape[1] == sum(self.dim_groups), 'input channels do not match.'
        shortcut = x
        input_list = torch.tensor_split(x, (self.dim_groups[2], self.dim_groups[1]+self.dim_groups[2]), dim=1)
        out_list = []
        out = self.vit_list[0](input_list[0])
        out_list.append(out)
        for i in range(1, len(self.dim_groups)):
            inp_list = out_list.copy()
            inp_list.append(input_list[i])
            inp = torch.cat(inp_list, dim=1)
            inp = self.conv_list[i-1](inp)
            out = self.vit_list[i](inp)
            out_list.append(out)
        x = torch.cat(out_list, dim=1)
        x = self.last_conv(x)
        x = self.channel_attn(x)
        x = shortcut + x
        return x

class HiSViT(nn.Module):
    def __init__(self, dim=[128, 256, 128], frames=8, size=[256,256], color_ch=1, blocks=8):
        super().__init__()
        self.color_ch = color_ch
        self.frames = frames
        self.size = size
        self.fem = RSTBWithInputConv(in_channels= 2,
                            kernel_size=(1, 3, 3),
                            groups=1,
                            num_blocks=1,
                            dim=dim[0],
                            input_resolution=[1, 256, 256],
                            depth=2,
                            num_heads=4,
                            window_size=[1, 8, 8],
                            mlp_ratio=2.,
                            qkv_bias=True, qk_scale=None,
                            use_checkpoint_attn=[False],
                            use_checkpoint_ffn=[False]
                            )

        self.dwsample = nn.Sequential(
                            nn.Conv3d(dim[0], dim[1], (1, 3, 3), (1, 2, 2), (0, 1, 1)),
                            nn.LeakyReLU(negative_slope=0.1, inplace=True)
                            )

        self.upsample = nn.Sequential(
                            nn.Conv3d(dim[1], 2 * dim[2], kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                            Rearrange('n c t h w -> n t c h w'),
                            nn.PixelShuffle(2),
                            Rearrange('n t c h w -> n c t h w'),
                            nn.LeakyReLU(negative_slope=0.1, inplace=True),
                            nn.Conv3d(dim[2]//2, dim[2], kernel_size=(1, 3, 3), padding=(0, 1, 1))
                            )   

        self.vrm = RSTBWithInputConv(in_channels= dim[0] + dim[2],
                            kernel_size=(1, 3, 3),
                            groups=1,
                            num_blocks=1,
                            dim=dim[2],
                            input_resolution=[1, 256, 256],
                            depth=2,
                            num_heads=4,
                            window_size=[1, 8, 8],
                            mlp_ratio=2.,
                            qkv_bias=True, qk_scale=None,
                            use_checkpoint_attn=[False],
                            use_checkpoint_ffn=[False]
                            )

        self.conv_list = nn.Sequential(
                            nn.Conv3d(dim[2], 64, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
                            nn.LeakyReLU(negative_slope=0.1, inplace=True), 
                            nn.Conv3d(64, self.color_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))
                            )

        self.resdnet_list = nn.ModuleList()
        for i in range(blocks):
            self.resdnet_list.append(GroupBlock(dim=dim[1], dim_groups=[dim[1]//4,dim[1]//4,dim[1]//2], reso=self.size[0]//2, frames=self.frames, idx=i))

    def forward(self, x):
        '''
        x  : b 2 t h w
        '''
        shallow_feat = self.fem(x.transpose(1,2).contiguous()).transpose(1,2).contiguous()
        out = self.dwsample(shallow_feat)
        for resdnet in self.resdnet_list:
            out = resdnet(out)
        out = self.upsample(out)
        out = torch.cat((shallow_feat, out), dim=1)

        out = self.vrm(out.transpose(1,2).contiguous()).transpose(1,2).contiguous()
        out = self.conv_list(out)
        
        return out
    

