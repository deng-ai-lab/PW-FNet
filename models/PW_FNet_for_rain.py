import torch
import torch.nn as nn
from typing import Type, Callable, Tuple, Optional, Set, List, Union
import torch.utils.checkpoint as checkpoint
import math
from torch import Tensor
from torchstat import stat
import torch.nn.functional as F
import numbers
import numpy as np
from matplotlib import pyplot as plt
from .LWN import _as_wavelet, get_filter_tensors, DWT, IDWT

from einops import rearrange

import pywt
import pywt.data
import torch.nn.functional as F


class PatchUnEmbed_for_upsample(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96, out_dim=64, kernel_size=None):
        super().__init__()
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        wt_type = 'db3'
        self.wavelet = _as_wavelet(wt_type)
        dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(
            wt_type, flip=True
        )
        self.rec_lo = nn.Parameter(rec_lo.flip(-1), requires_grad=True)
        self.rec_hi = nn.Parameter(rec_hi.flip(-1), requires_grad=True)

        self.waverec = IDWT(self.rec_lo, self.rec_hi, wavelet=wt_type, level=1)

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_dim * patch_size ** 2, kernel_size=1),
        )

    def forward(self, x):
        x = self.proj(x)
        ya, yh, yv, yd = torch.chunk(x, 4, dim=1)
        x = self.waverec([ya, (yh, yv, yd)], None)
        return x


class DownSample(nn.Module):
    """
    DownSample: Conv
    B*H*W*C -> B*(H/2)*(W/2)*(2*C)
    """

    def __init__(self, input_dim, output_dim, patch_size=2):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = output_dim

        wt_type = 'db3'
        self.wavelet = _as_wavelet(wt_type)
        dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(
            wt_type, flip=True
        )
        self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)
        self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)

        self.wavedec = DWT(self.dec_lo, self.dec_hi, wavelet=wt_type, level=1)

        self.proj = nn.Sequential(
            nn.Conv2d(input_dim * patch_size ** 2, input_dim * 2, kernel_size=1))

    def forward(self, x):
        ya, (yh, yv, yd) = self.wavedec(x)
        x = torch.cat([ya, yh, yv, yd], dim=1)
        x = self.proj(x)
        return x


class OurFFN(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(OurFFN, self).__init__()
        self.dim = dim
        self.dim_sp = dim * 2
        # PW first or DW first?
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU()
        )

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1,
                      groups=self.dim_sp),
        )

        self.gelu = nn.GELU()
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
        )


    def forward(self, x):
        x = self.conv_init(x)
        x = self.conv1_1(x)
        x = self.gelu(x)
        x = self.conv_fina(x)

        return x


class OurTokenMixer_For_Local(nn.Module):
    def __init__(
            self,
            dim
    ):
        super(OurTokenMixer_For_Local, self).__init__()
        self.dim = dim
        self.dim_sp = dim
        self.CDilated = nn.Sequential(nn.BatchNorm2d(self.dim_sp),
                                     nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=1, groups=self.dim_sp)
                                        )

    def forward(self, x):
        x = self.CDilated(x)  # (batch, c*2, h, w/2+1)

        return x

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.dim = in_channels

        self.conv_layer = nn.Sequential(
                                        nn.BatchNorm2d(out_channels * 2),
                                        nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                                        kernel_size=1, stride=1, padding=0, groups=self.groups,bias=True),
                                        nn.GELU(),
                                        )


    def forward(self, x):
        batch, c, h, w = x.size()
        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        # (batch, c, 2, h, w/2+1)
        ffted = rearrange(ffted, 'b c h w d -> b (c d) h w').contiguous()
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = rearrange(ffted, 'b (c d) h w -> b c h w d', d=2).contiguous()
        ffted = torch.view_as_complex(ffted)

        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output


class OurTokenMixer_For_Gloal(nn.Module):
    def __init__(
            self,
            dim
    ):
        super(OurTokenMixer_For_Gloal, self).__init__()
        self.dim = dim
        # PW first or DW first?
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU()
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1)
        )
        self.FFC = FourierUnit(self.dim*2, self.dim*2)

    def forward(self, x):
        x = self.conv_init(x)
        x = self.FFC(x)
        x = self.conv_fina(x)

        return x


class OurMixer(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer_for_local=OurTokenMixer_For_Local,
            token_mixer_for_gloal=OurTokenMixer_For_Gloal
    ):
        super(OurMixer, self).__init__()
        self.dim = dim
        self.mixer_local = token_mixer_for_local(dim=self.dim)
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim)

        self.ca_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
        )

        self.gelu = nn.GELU()
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.GELU()
        )


    def forward(self, x):
        x = self.conv_init(x)
        x = self.mixer_gloal(x)
        x = self.gelu(x)
        x = self.ca_conv(x)

        return x


class OurBlock(nn.Module):
    def __init__(
            self,
            dim,
            norm_layer=nn.BatchNorm2d,
            token_mixer=OurMixer
    ):
        super(OurBlock, self).__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mixer = token_mixer(dim=self.dim)
        self.ffn = OurFFN(dim=self.dim)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        copy = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = x * self.beta + copy

        copy = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x * self.gamma + copy

        return x


# need drop_path?
class OurStage(nn.Module):
    def __init__(
            self,
            depth=int,
            in_channels=int,
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(OurStage, self).__init__()
        # Init blocks
        self.blocks = nn.Sequential(*[
                OurBlock(
                    dim=in_channels,
                    norm_layer=nn.BatchNorm2d,
                    token_mixer=OurMixer
                )
            for index in range(depth)
        ])

    def forward(self, input=torch.Tensor) -> torch.Tensor:
        output = self.blocks(input)
        return output

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = nn.Sequential(
            nn.Conv2d(channel*2, channel, 1),
            nn.Conv2d(channel, channel, 3, 1, 1, groups=channel)
        )

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))


class SCM(nn.Module):
    def __init__(self, dim, factor):
        super(SCM, self).__init__()
        self.sp = nn.Sequential(
            nn.Conv2d(3*factor**2, dim//2, 1),
            nn.Conv2d(dim//2, dim//2, 3, 1, 1, groups=dim//2),
            nn.GELU(),
            nn.Conv2d(dim//2, dim, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.GELU()
        )

        self.fr = nn.Sequential(
            nn.Conv2d(6*factor**2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )

        self.agg = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x_sp = self.sp(x)
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        ffted = rearrange(ffted, 'b c h w d -> b (c d) h w').contiguous()
        ffted = self.fr(ffted)  # (batch, c*2, h, w/2+1)
        ffted = rearrange(ffted, 'b (c d) h w -> b c h w d', d=2).contiguous()
        ffted = torch.view_as_complex(ffted)
        x_fr = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        x = self.agg(torch.cat([x_sp, x_fr], dim=1))

        return x


class DCM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DCM, self).__init__()
        self.sp = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        )


    def forward(self, x):
        x = self.sp(x)

        return x


class Backbone_new(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, patch_size=1,
                 embed_dim=[48, 96, 192, 96, 48], depth=[2, 2, 2, 2, 2],
                 embed_kernel_size=3,
                 downsample_kernel_size=None, upsample_kernel_size=None):
        super(Backbone_new, self).__init__()

        self.patch_size = patch_size
        if downsample_kernel_size is None:
            downsample_kernel_size = 4
        if upsample_kernel_size is None:
            upsample_kernel_size = 4

        self.patch_embed = SCM(embed_dim[0], 1)
        self.layer1 = OurStage(depth=depth[0], in_channels=embed_dim[0])
        self.skip1 = nn.Conv2d(embed_dim[1], embed_dim[0], 1)
        self.downsample1 = DownSample(input_dim=embed_dim[0], output_dim=embed_dim[1])
        self.layer2 = OurStage(depth=depth[1], in_channels=embed_dim[1])
        self.skip2 = nn.Conv2d(embed_dim[2], embed_dim[1], 1)
        self.downsample2 = DownSample(input_dim=embed_dim[1], output_dim=embed_dim[2])
        self.layer3 = OurStage(depth=depth[2], in_channels=embed_dim[2])
        self.upsample3 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[6],
                                                   out_dim=embed_dim[7])
        self.layer8 = OurStage(depth=depth[7], in_channels=embed_dim[7])
        self.upsample4 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[7],
                                                   out_dim=embed_dim[8])
        self.layer9 = OurStage(depth=depth[8], in_channels=embed_dim[8])
        self.patch_unembed = DCM(embed_dim[0], out_chans)


        self.FAM1 = FAM(embed_dim[1])
        self.SCM1 = SCM(embed_dim[1], 2)
        self.FAM2 = FAM(embed_dim[2])
        self.SCM2 = SCM(embed_dim[2], 4)
        self.FAM3 = FAM(embed_dim[1])
        self.SCM3 = SCM(embed_dim[1], 2)
        self.FAM4 = FAM(embed_dim[0])
        self.SCM4 = SCM(embed_dim[0], 1)
        self.decoder1 = DCM(embed_dim[0], out_chans)
        self.decoder2 = DCM(embed_dim[1], out_chans*4)
        self.decoder3 = DCM(embed_dim[2], out_chans*16)
        self.decoder4 = DCM(embed_dim[1], out_chans*4)

        wt_type = 'db3'
        self.wavelet = _as_wavelet(wt_type)
        dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(
            wt_type, flip=True
        )
        self.dec_lo = nn.Parameter(dec_lo, requires_grad=False)
        self.dec_hi = nn.Parameter(dec_hi, requires_grad=False)
        self.rec_lo = nn.Parameter(rec_lo.flip(-1), requires_grad=False)
        self.rec_hi = nn.Parameter(rec_hi.flip(-1), requires_grad=False)

        self.wavedec = DWT(self.dec_lo, self.dec_hi, wavelet=wt_type, level=1)
        self.waverec = IDWT(self.rec_lo, self.rec_hi, wavelet=wt_type, level=1)


    def forward(self, x):
        ya, (yh, yv, yd) = self.wavedec(x)
        x_2 = torch.cat([ya, yh, yv, yd], dim=1)

        copy0 = x
        x = self.patch_embed(x)
        x = self.layer1(x)
        e1 = self.decoder1(x) + copy0
        copy1 = x

        x = self.downsample1(x)
        ez1 = self.SCM1(x_2)
        x = self.FAM1(x, ez1)
        x = self.layer2(x)
        e2 = self.decoder2(x) + x_2

        ez3 = self.SCM3(x_2)
        x = self.FAM3(x, ez3)
        x = self.layer8(x)
        e4 = self.decoder4(x) + x_2
        x = self.upsample4(x)

        x = self.skip1(torch.cat([x, copy1], dim=1))
        ez4 = self.SCM4(copy0)
        x = self.FAM4(x, ez4)
        x = self.layer9(x)
        x = self.patch_unembed(x) + copy0

        return [e1, e2, e4, x]


def PW_FNet_8448():
    return Backbone_new(
        patch_size=1,
        embed_dim=[32, 64, 128, 256, 512, 256, 128, 64, 32],
        depth=[8, 4, 2, 0, 0, 0, 0, 4, 8],
        embed_kernel_size=3
    )

def PW_FNet_6446():
    return Backbone_new(
        patch_size=1,
        embed_dim=[32, 64, 128, 256, 512, 256, 128, 64, 32],
        depth=[6, 4, 2, 0, 0, 0, 0, 4, 6],
        embed_kernel_size=3
    )




