import os
import math
import sys
from typing import Optional
from ptflops import get_model_complexity_info
import numpy as np
import pytorch_msssim
import cv2
import time

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import torchvision.transforms as transforms


from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import _assert, trunc_normal_, to_2tuple, DropPath, Mlp
from timm.models.fx_features import register_notrace_function

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.layers import BitEstimator


def lower_bound_fwd(x: Tensor, bound: Tensor) -> Tensor:
    return torch.max(x, bound)

def lower_bound_bwd(x: Tensor, bound: Tensor, grad_output: Tensor):
    pass_through_if = (x >= bound) | (grad_output < 0)
    return pass_through_if * grad_output, None

class LowerBoundFunction(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return lower_bound_fwd(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        return lower_bound_bwd(x, bound, grad_output)

class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.

    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """

    bound: Tensor

    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)

class NonNegativeParametrizer(nn.Module):
    """
    Non negative reparametrization.

    Used for stability during training.
    """

    pedestal: Tensor

    def __init__(self, minimum: float = 0, reparam_offset: float = 2**-18):
        super().__init__()

        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)

        pedestal = self.reparam_offset**2
        self.register_buffer("pedestal", torch.Tensor([pedestal]))
        bound = (self.minimum + self.reparam_offset**2) ** 0.5
        self.lower_bound = LowerBound(bound)

    def init(self, x: Tensor) -> Tensor:
        return torch.sqrt(torch.max(x + self.pedestal, self.pedestal))

    def forward(self, x: Tensor) -> Tensor:
        out = self.lower_bound(x)
        out = out**2 - self.pedestal
        return out
    
class GDN(nn.Module):
    r"""Generalized Divisive Normalization layer.

    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).

    .. math::

       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}

    """

    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x**2, gamma, beta)

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out


def get_relative_position_index(win_h, win_w):  # [WH*WW, WH*WW]
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)], indexing='ij'))  # 2, WH*WW, WH*WW
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, WH*WW, WH*WW (xaxis matrix & yaxis matrix)
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # WH*WW, WH*WW, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # WH*WW, WH*WW

def window_partition(x, window_size: int):  # [B, ph, pw, D] -> [B*wh*ww, WH, WW, D], (wh, ww)
    """
    Args:
        x: [B, ph, pw, D]
        window_size (int): The height and width of the window.
    Returns:
        [B*wh*ww, WH, WW, D], (wh, ww), where wh, ww are the number of the window, and wh=ph//WH, ww=pw//WW
    """
    B, ph, pw, D = x.size()
    wh, ww = ph//window_size, pw//window_size # number of windows (height, width)
    if 0 in [wh, ww]:
        # if feature map size is smaller than window size, do not partition
        return x, (wh, ww)
    windows = rearrange(x, 'b (h wh) (w ww) c -> (b h w) wh ww c', wh=window_size, ww=window_size).contiguous()
    return windows.contiguous(), (wh, ww)
    
@register_notrace_function  # reason: int argument is a Proxy
def window_unpartition(windows, num_windows):   # [B*wh*ww, WH, WW, D] -> [B, ph, pw, D]
    """
    Args:
        windows: [B*wh*ww, WH, WW, D]
        num_windows (tuple[int]): The height and width of the window.
    Returns:
        x: [B, ph, pw, D]
    """
    x = rearrange(windows, '(p h w) wh ww c -> p (h wh) (w ww) c', h=num_windows[0], w=num_windows[1])
    return x.contiguous()

class WindowAttention(nn.Module):   # [B*wh*ww, WH*WW, D] -> [B*wh*ww, WH*WW, D]
    def __init__(self, dim, num_heads, window_size, head_dim=None, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)   # (WH, WW)
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)  # [num_heads, 1, 1], 是一个可训练的参数，用于缩放attention map
        
        # define a parameter table of relative position bias, shape: 2*WH-1 * 2*WW-1, num_heads
        # 因此，相对位置偏置表的形状为 (2 * win_h - 1) * (2 * win_w - 1) 行和 num_heads 列，其中每行对应于一个相对位置偏置向量，
        # 每列对应于一个注意力头。这样，模型可以利用这个表来学习不同位置之间的相对位置关系，并在自注意力计算中进行调整和加权。
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win_h - 1) * (2 * win_w - 1), num_heads))     # [2*WH-1 * 2*WW-1, num_heads], 是一个可训练的参数，用于计算相对位置偏置
        
        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(win_h, win_w))  # [WH*WW, WH*WW]
        
        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)  # dim -> attn_dim * 3
        self.attn_drop = nn.Dropout(attn_drop)                  # atten_drop_dropout
        self.proj = nn.Linear(attn_dim, dim)                    # attn_dim -> dim
        self.proj_drop = nn.Dropout(proj_drop)                  # proj_drop_dropout
        
        trunc_normal_(self.relative_position_bias_table, std=.02)   # 对张量进行截断正态分布初始化。对 relative_position_bias_table 进行截断正态分布初始化是为了为模型提供一个合适的起点，让模型能够学习并利用相对位置偏置来提高其性能和表达能力。
        self.softmax = nn.Softmax(dim=-1)
        
    def _get_rel_pos_bias(self) -> torch.Tensor:    # 1, num_heads, WH*WW, WH*WW 
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # WH*WW, WH*WW, num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, WH*WW, WH*WW
        return relative_position_bias.unsqueeze(0)  # 1, num_heads, WH*WW, WH*WW
    
    def forward(self, x, mask: Optional[torch.Tensor] = None):

        B_, N, C = x.shape  # [B*wh*ww, WH*WW, D] 把窗口中的元素个数看作序列长度，把D看作序列维度
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4) 
        # [B*wh*ww, WH*WW, D] -> [B*wh*ww, WH*WW, attn_dim * 3] -> [B*wh*ww, WH*WW, 3, num_heads, dim_per_head] ->[qkv(3), B*wh*ww, nheads, WH*WW, dim_per_head]
        q, k, v = qkv.unbind(0)  # [B*wh*ww, nheads, WH*WW, dim_per_head]
        
        # scaled cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) # [B*wh*ww, nheads, WH*WW, WH*WW]，使用L2范数对q和k进行归一化(好处：尺度不影响度量；计算数值稳定性)，然后计算q和k的转置矩阵的乘积，得到注意力图
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()  # [num_heads, 1, 1]，对logit_scale进行截断，然后取指数，得到一个缩放因子
        attn = attn * logit_scale   # [B*wh*ww, nheads, WH*WW, WH*WW]
        
        attn = attn + self._get_rel_pos_bias()  # [B*wh*ww, nheads, WH*WW, WH*WW]
        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(B_ // num_win, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)   # [B*wh*ww, nheads, WH*WW, WH*WW]

        attn = self.attn_drop(attn)     # [B*wh*ww, nheads, WH*WW, WH*WW]
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1) # [B*wh*ww, nheads, WH*WW, WH*WW] -> [B*wh*ww, nheads, WH*WW, dim_per_head] -> [B*wh*ww, WH*WW, nheads, dim_per_head] -> [B*wh*ww, WH*WW, nheads*dim_per_head]=[B*wh*ww, WH*WW, D]
        x = self.proj(x)        # [B*wh*ww, WH*WW, atten_d] -> [B*wh*ww, WH*WW, D]
        x = self.proj_drop(x)   # [B*wh*ww, WH*WW, D]
        return x
    

class SWA(nn.Module):    # [B, ph, pw D] -> [B, wh, ww, 1, 1, D]

    def __init__(self, dim, window_size, ngram, ngram_num_heads, padding_mode='seq_refl_win_pad'):
        super().__init__()
        
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.ngram = ngram
        self.padding_mode = padding_mode
        
        self.unigram_embed = nn.Conv2d(dim, dim//2,
                                       kernel_size=(self.window_size[0], self.window_size[1]), 
                                       stride=self.window_size, padding=0, groups=dim//2)       # [B, D, ph, pw] -> [B, D/2, wh, ww], 输入输出被分成groups, 分别进行卷积，然后再concat，为了减少参数量
        self.ngram_attn = WindowAttention(dim=dim//2, num_heads=ngram_num_heads, window_size=ngram) 
        self.avg_pool = nn.AvgPool2d(ngram)
        self.merge = nn.Conv2d(dim, dim, 1, 1, 0)
        
    def seq_refl_win_pad(self, x, back=False):  # [B, D/2, wh, ww] -> [B, D/2, wh+ngram-1, ww+ngram-1]
        if self.ngram == 1: return x        # if ngram==1, no need to pad
        x = TF.pad(x, (0,0,self.ngram-1,self.ngram-1)) if not back else TF.pad(x, (self.ngram-1,self.ngram-1,0,0))  #
        if self.padding_mode == 'zero_pad':
            return x                      # if padding_mode=='zero_pad', no need to change
        if not back:
            (start_h, start_w), (end_h, end_w) = to_2tuple(-2*self.ngram+1), to_2tuple(-self.ngram)
            # pad lower
            x[:,:,-(self.ngram-1):,:] = x[:,:,start_h:end_h,:]
            # pad right
            x[:,:,:,-(self.ngram-1):] = x[:,:,:,start_w:end_w]
        else:
            (start_h, start_w), (end_h, end_w) = to_2tuple(self.ngram), to_2tuple(2*self.ngram-1)
            # pad upper
            x[:,:,:self.ngram-1,:] = x[:,:,start_h:end_h,:]
            # pad left
            x[:,:,:,:self.ngram-1] = x[:,:,:,start_w:end_w]
            
        return x
    
    def sliding_window_attention_old(self, unigram): #unigram: [B, D/2, wh, ww], -> [B, D/2, (wh-ngram+1), (ww-ngram+1)]
        slide = unigram.unfold(3, self.ngram, 1).unfold(2, self.ngram, 1)   # [B, D/2, wh+ngram-1, ww+ngram-1, ngram, ngram]
        # 首先在高度维度上进行展开，展开后的维度为 [B, D/2, wh, wh-ngram+1, ngram]，其中 wh-ngram+1 为展开后的高度，ngram 为展开后的高度维度上的窗口大小，1 为展开后的高度维度上的步长。
        # 然后在宽度维度上进行展开，展开后的维度为 [B, D/2, wh-ngram+1, ww-ngram+1, ngram, ngram]，其中 ww-ngram+1 为展开后的宽度，ngram 为展开后的宽度维度上的窗口大小，1 为展开后的宽度维度上的步长。
        slide = rearrange(slide, 'b c h w ww hh -> b (h hh) (w ww) c') # [B, (wh-ngram+1)*ngram, (ww-ngram+1)*ngram, D/2]
        slide, num_windows = window_partition(slide, self.ngram) # [B*(wh-ngram+1)*(ww-ngram+1), ngram*ngram, D/2], ((wh-ngram+1), (ww-ngram+1))
        slide = slide.view(-1, self.ngram*self.ngram, self.dim//2) # [B*(wh-ngram+1)*(ww-ngram+1), ngram*ngram, D/2]
        
        context = self.ngram_attn(slide).view(-1, self.ngram, self.ngram, self.dim//2) # [B*(wh-ngram+1)*(ww-ngram+1), ngram*ngram, D/2]
        context = window_unpartition(context, num_windows) # [B, (wh-ngram+1)*ngram, (ww-ngram+1)*ngram, D/2]
        context = rearrange(context, 'b h w d -> b d h w') # [B, D/2, (wh-ngram+1)*ngram, (ww-ngram+1)*ngram]
        context = self.avg_pool(context) # [B, D/2, (wh-ngram+1), (ww-ngram+1)]
        return context      # [B, D/2, (wh-ngram+1), (ww-ngram+1)]

    def sliding_window_attention(self, unigram): #unigram: [B, D/2, wh, ww] -> [B, D/2, (wh-ngram+1), (ww-ngram+1)]
        _, _, wh, ww = unigram.size()
        unigram = F.unfold(unigram, kernel_size=self.ngram, dilation=1, stride=1)   # [B, D/2* ngram * ngram, L] L=(wh−ngram+1)×(ww−ngram+1)
        b, c_kh_kw, _ = unigram.size()
        slice = unigram.view(b, c_kh_kw, wh-self.ngram+1, -1)   #[B, D/2* ngram * ngram, (wh−ngram+1), (ww−ngram+1)]
        slice_1 = slice.permute(0, 2, 3, 1).contiguous()       # [B, (wh−ngram+1), (ww−ngram+1), D/2*ngram*ngram]
        slice_2 = slice_1.view(b, wh-self.ngram+1, ww-self.ngram+1, -1, self.ngram, self.ngram)     # [B, (wh−ngram+1), (ww−ngram+1), D/2, ngram, ngram]
        # print(slice)

        slice_3 = rearrange(slice_2, 'b h w c ww hh -> b (h hh) (w ww) c')  # [B, (wh−ngram+1)*ngram, (ww−ngram+1)*ngram, D/2]
        slice_4, num_windows = window_partition(slice_3, self.ngram)    # [B*(wh-ngram+1)*(ww-ngram+1), ngram*ngram, D/2], ((wh-ngram+1), (ww-ngram+1))

        slice_5 = slice_4.view(-1, self.ngram*self.ngram, self.dim//2) # [B*(wh-ngram+1)*(ww-ngram+1), ngram*ngram, D/2]

        context = self.ngram_attn(slice_5).view(-1, self.ngram, self.ngram, self.dim//2) # [B*(wh-ngram+1)*(ww-ngram+1), ngram*ngram, D/2] -> [B*(wh-ngram+1)*(ww-ngram+1), ngram, ngram, D/2]
        context = window_unpartition(context, num_windows) # [B, (wh-ngram+1)*ngram, (ww-ngram+1)*ngram, D/2]
        context = rearrange(context, 'b h w d -> b d h w') # [B, D/2, (wh-ngram+1)*ngram, (ww-ngram+1)*ngram]
        context = self.avg_pool(context) # [B, D/2, (wh-ngram+1), (ww-ngram+1)]
        return context      # [B, D/2, (wh-ngram+1), (ww-ngram+1)]

    def forward(self, x):
        B, ph, pw, D = x.size()
        x = rearrange(x, 'b ph pw d -> b d ph pw') # [B, D, ph, pw]
        unigram = self.unigram_embed(x) # [B, D/2, wh, ww]
        
        unigram_forward_pad = self.seq_refl_win_pad(unigram, False) # [B, D/2, wh+ngram-1, ww+ngram-1]
        unigram_backward_pad = self.seq_refl_win_pad(unigram, True) # [B, D/2, wh+ngram-1, ww+ngram-1]
        
        context_forward = self.sliding_window_attention(unigram_forward_pad) # [B, D/2, wh, ww]
        context_backward = self.sliding_window_attention(unigram_backward_pad) # [B, D/2, wh, ww]
        
        context_bidirect = torch.cat([context_forward, context_backward], dim=1) # [B, D, wh, ww]
        context_bidirect = self.merge(context_bidirect) # [B, D, wh, ww]
        context_bidirect = rearrange(context_bidirect, 'b d h w -> b h w d') # [B, wh, ww, D]
        
        return context_bidirect.unsqueeze(-2).unsqueeze(-2).contiguous() # [B, wh, ww, 1, 1, D] 
    

class NWA(nn.Module):   # [B, ph, pw, D] -> [B*wh*ww, WH, WW, D], (wh, ww)

    def __init__(self, dim, window_size, ngram, ngram_num_heads, shift_size=0):
        super().__init__()
        self.window_size = window_size
        self.ngram = ngram
        self.shift_size = shift_size
        
        self.ngram_context = SWA(dim, window_size, ngram, ngram_num_heads, padding_mode='seq_refl_win_pad')
    
    def forward(self, x):
        B, ph, pw, D = x.size()
        wh, ww = ph//self.window_size, pw//self.window_size # number of windows (height, width)
        _assert(0 not in [wh, ww], "feature map size should be larger than window size!")
        
        context = self.ngram_context(x) # [B, wh, ww, 1, 1, D]
        
        windows = rearrange(x, 'b (h wh) (w ww) c -> b h w wh ww c', 
                            wh=self.window_size, ww=self.window_size).contiguous() # [B, wh, ww, WH, WW, D]. semi window partitioning
        windows+=context # [B, wh, ww, WH, WW, D]. inject context
        
        # Cyclic Shift
        if self.shift_size>0:
            x = rearrange(windows, 'b h w wh ww c -> b (h wh) (w ww) c').contiguous() # [B, ph, pw, D]. re-patchfying
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) # [B, ph, pw, D]. cyclic shift
            windows = rearrange(shifted_x, 'b (h wh) (w ww) c -> b h w wh ww c', 
                                wh=self.window_size, ww=self.window_size).contiguous() # [B, wh, ww, WH, WW, D]. re-semi window partitioning
            
        windows = rearrange(windows, 'b h w wh ww c -> (b h w) wh ww c').contiguous() # [B*wh*ww, WH, WW, D]. window partitioning
        
        return windows, (wh, ww)


class ESTB(nn.Module): # [B, ph*pw, D], (ph, pw) -> [B, ph*pw, D], (ph, pw)

    def __init__(
        self, dim, ngram, num_heads, window_size, shift_size,
        head_dim=None, mlp_ratio=4., qkv_bias=True, 
        drop=0., attn_drop=0., drop_path=0.,
        act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        
        super().__init__()
        self.dim = dim

        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        _assert(0 <= self.shift_size < self.window_size, "shift_size must in 0~window_size")
        
        self.ngram_window_partition = NWA(dim, window_size, ngram, num_heads, shift_size=shift_size)    # [B, ph, pw, D] -> [B*wh*ww, WH, WW, D], (wh, ww)
        self.attn = WindowAttention(
            dim, num_heads=num_heads, head_dim=head_dim, window_size=to_2tuple(self.window_size),
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        
    def make_mask(self, num_patches):
        ph, pw = num_patches
        img_mask = torch.zeros((1, ph, pw, 1)) # [1, ph, pw, 1]
        cnt = 0
        for h in (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)):
            for w in (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows, (wh,ww) = window_partition(img_mask, self.window_size)  # [wh*ww*1, WH, WW, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # [wh*ww, WH*WW]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # [wh*ww, WH, WW]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0)) # [wh*ww, WH, WW]
        return attn_mask
    
    def _attention(self, x, num_patches):   # [B, ph*pw, D], (ph, pw) -> [B, ph*pw, D], (ph, pw)
        # window partition - (cyclic shift) - cosine attention - window unpartition - (reverse shift)
        ph, pw = num_patches
        B, p, D = x.size()
        _assert(p == ph * pw, f"size is wrong!")
        
        x = x.view(B, ph, pw, D) # [B, ph, pw, D], Unembedding
        
        # N-Gram Window Partition (-> cyclic shift)
        x_windows, (wh,ww) = self.ngram_window_partition(x) # [B*wh*ww, WH, WW, D], (wh, ww)
        
        x_windows = x_windows.view(-1, self.window_size * self.window_size, D)  # [B*wh*ww, WH*WW, D], Re-embedding
        
        # W-MSA/SW-MSA
        attn_mask = self.make_mask(num_patches).to(x.device) if self.shift_size>0 else None
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [B*wh*ww, WH*WW, D]
        
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, D) # [B*wh*ww, WH, WW, D], Unembedding
        
        # Window Unpartition
        shifted_x = window_unpartition(attn_windows, (wh,ww))  # [B, ph, pw, D]
        
        # Reverse Cyclic Shift
        reversed_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) if self.shift_size > 0 else shifted_x # [B, ph, pw, D]
        reversed_x = reversed_x.view(B, ph*pw, D) # [B, ph*pw, D], Re-embedding
        
        return reversed_x

    def forward(self, x, num_patches):
        x_ = x
        # (S)W Attention -> Layer-Norm -> Drop-Path -> Skip-Connection
        x = x + self.drop_path(self.norm1(self._attention(x, num_patches))) # [B, ph*pw, D]
        # FFN -> Layer-Norm -> Drop-Path -> Skip-Connection
        x = x + self.drop_path(self.norm2(self.ffn(x))) # [B, ph*pw, D]
        # return x_, x, num_patches
        return x, num_patches


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module: # [B, C_in, H, W] -> [B, C_out, H, W]
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module: # [B, C_in, H, W] -> [B, C_out, H, W]
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:   # [B, C_in, H, W] -> [B, C_out, H*r, W*r]
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r**2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )

class ResidualBlock(nn.Module): # [B, C_in, H, W] -> [B, C_out, H, W]


    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out

class ResidualBlockWithStride(nn.Module):  # [B, C_in, H, W] -> [B, C_out, H//2, W//2] if stride = 2


    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = GDN(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out

class ResidualBlockUpsample(nn.Module):     # [B, C_in, H, W] -> [B, C_out, H*upsample, W*upsample]


    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out


class ESTCB(nn.Module): # [B, C_conv + C_eswint, H, W] -> [B, C_conv + C_eswint, H, W]
    def __init__(self, conv_dim, swint_dim, ngram, num_heads, window_size, shift_size, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.conv_dim = conv_dim
        self.swint_dim = swint_dim
        self.ngram = ngram
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path = drop_path

        self.eswint_block = ESTB(dim=self.swint_dim,
                                    ngram=self.ngram, 
                                    num_heads=self.num_heads, 
                                    window_size=self.window_size, 
                                    shift_size=self.shift_size, 
                                    drop=self.drop, 
                                    attn_drop=self.attn_drop, 
                                    drop_path=self.drop_path)   # [B, ph*pw, D], (ph, pw) -> [B, ph*pw, D], (ph, pw)
        
        self.feature_merge = nn.Conv2d(self.conv_dim+self.swint_dim, 
                                 self.conv_dim+self.swint_dim, 
                                 kernel_size=1, stride=1, padding=0, bias=True)
    
        self.conv_block = ResidualBlock(self.conv_dim, self.conv_dim)

    def forward(self, x):   # x: [B, C_conv + C_eswint, H, W]

        x_conv, x_swint = torch.split(self.feature_merge(x), (self.conv_dim, self.swint_dim), dim=1)    # [B, C_conv, H, W], [B, C_eswint, H, W]

        x_conv = self.conv_block(x_conv)        # [B, C_conv, H, W]

        _, _, patch_h, patch_w = x_swint.size()
        num_patches = (patch_h, patch_w)
        x_swint_1 = Rearrange('b c h w -> b (h w) c')(x_swint)
        x_swint_2, num_patches = self.eswint_block(x_swint_1, num_patches)
        x_swint_3 = Rearrange('b (h w) c -> b c h w', h=num_patches[0], w=num_patches[1])(x_swint_2)    # [B, C_eswint, H, W]
        estc_x = torch.cat([x_conv, x_swint_3], dim=1)    # [B, C_conv + C_eswint, H, W]
        x_out = self.feature_merge(estc_x)      # [B, C_conv + C_eswint, H, W]

        x_out = x + x_out   # [B, C_conv + C_eswint, H, W]

        return x_out



class AttentionBlock(nn.Module):    # [B, C, H, W] -> [B, C, H, W]
    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out

class ESTC_CHARM_Block(AttentionBlock): # [B, C=320+i*slice, H, W] -> [B, C=320+i*slice, H, W]
    def __init__(self, in_dim, out_dim, ngram, window_size, num_heads, drop=0.1, attn_drop=0.1, drop_path=0.1, bottleneck_dim=192):
        super().__init__(N=bottleneck_dim)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ngram = ngram
        self.window_size = window_size
        self.num_heads = num_heads
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path = drop_path
        self.bottleneck_dim = bottleneck_dim
        self.shift_size = [0, window_size // 2]

        self.ngswint_1 = ESTB(dim=self.bottleneck_dim, ngram=self.ngram, num_heads=self.num_heads, window_size=self.window_size,
                     shift_size=self.shift_size[0], drop=self.drop, attn_drop=self.attn_drop, 
                     drop_path=self.drop_path)
        self.ngswint_2 = ESTB(dim=self.bottleneck_dim, ngram=self.ngram, num_heads=self.num_heads, window_size=self.window_size,
                     shift_size=self.shift_size[1], drop=self.drop, attn_drop=self.attn_drop, 
                     drop_path=self.drop_path)

        self.in_conv = conv1x1(self.in_dim, self.bottleneck_dim)
        self.out_conv = conv1x1(self.bottleneck_dim, self.out_dim)

    def forward(self, x):

        # x: [B, C=320+i*slice, H // 16, W // 16] -> [B, C=192, H // 16, W // 16]

        x = self.in_conv(x)  # [B, C=320+i*slice, H, W] -> [B, C=192, H, W]
        identity = x

        ph, pw = x.size(2), x.size(3)
        ngswint_in = Rearrange('b c h w -> b (h w) c')(x)   # [B, H*W, C=192]
        # x_out, patch_num = self.ngswint(ngswint_in, patch_num)
        x_out_1, patch_num = self.ngswint_1(ngswint_in, (ph, pw))
        x_out, patch_num = self.ngswint_2(x_out_1, patch_num)

        x_out = Rearrange('b (h w) c -> b c h w', h=patch_num[0])(x_out)    # [B, C=192, H, W]

        branch_1 = self.conv_a(x)
        branch_2 = self.conv_b(x_out)
        atten_ch = torch.sigmoid(branch_2)
        res = branch_1 * atten_ch               # [B, C=192, H, W]

        x_ch_atten = identity + res             # [B, C=192, H, W]

        out = self.out_conv(x_ch_atten)         # [B, C=192, H, W] -> [B, C=320+i*slice, H, W]

        return out


class Checkerboard_Window_Attention(nn.Module):   # [B*wh*ww, WH*WW, D] -> [B*wh*ww, WH*WW, D]

    def __init__(self, dim, num_heads, window_size, head_dim=None, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)   # (WH, WW)
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)  # [num_heads, 1, 1], 是一个可训练的参数，用于缩放attention map
        
        # define a parameter table of relative position bias, shape: 2*WH-1 * 2*WW-1, num_heads
        # 因此，相对位置偏置表的形状为 (2 * win_h - 1) * (2 * win_w - 1) 行和 num_heads 列，其中每行对应于一个相对位置偏置向量，
        # 每列对应于一个注意力头。这样，模型可以利用这个表来学习不同位置之间的相对位置关系，并在自注意力计算中进行调整和加权。
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win_h - 1) * (2 * win_w - 1), num_heads))     # [2*WH-1 * 2*WW-1, num_heads], 是一个可训练的参数，用于计算相对位置偏置
        
        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(win_h, win_w))  # [WH*WW, WH*WW]
        
        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)  # dim -> attn_dim * 3
        self.attn_drop = nn.Dropout(attn_drop)                  # atten_drop_dropout
        self.proj = nn.Linear(attn_dim, dim)                    # attn_dim -> dim
        self.proj_drop = nn.Dropout(proj_drop)                  # proj_drop_dropout
        
        trunc_normal_(self.relative_position_bias_table, std=.02)   # 对张量进行截断正态分布初始化。对 relative_position_bias_table 进行截断正态分布初始化是为了为模型提供一个合适的起点，让模型能够学习并利用相对位置偏置来提高其性能和表达能力。
        self.softmax = nn.Softmax(dim=-1)
        
    def _get_rel_pos_bias(self) -> torch.Tensor:    # 1, num_heads, WH*WW, WH*WW 
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # WH*WW, WH*WW, num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, WH*WW, WH*WW
        return relative_position_bias.unsqueeze(0)  # 1, num_heads, WH*WW, WH*WW

    @staticmethod
    def checkerboard_attn_mask(win_size, device):  # win_size=(WH,WW) win_mask=(1, 1, WH*WW, WH*WW) used for mask the attention map
        wh, ww = win_size
        win_mask = np.ones((1, 1, wh, ww))
        win_mask[:, :, 0::2, 1::2] = 0
        win_mask[:, :, 1::2, 0::2] = 0
        win_mask_row = win_mask.reshape(1, -1)
        win_mask_col = win_mask.reshape(-1, 1)
        win_mask = np.dot(win_mask_col, win_mask_row)
        win_mask = torch.tensor(win_mask, device=device)
        win_mask = win_mask.bool()

        return win_mask 

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [B*wh*ww, WH*WW, D]
            mask: (0/-inf) mask with shape of (wh*ww, WH*WW, WH*WW) or None
        Returns:
            x: [B*wh*ww, WH*WW, D]
        """
        B_, N, C = x.shape  # x = [B*wh*ww, WH*WW, D] 把窗口中的元素个数看作序列长度，把D看作序列维度
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4) 
        # [B*wh*ww, WH*WW, D] -> [B*wh*ww, WH*WW, attn_dim * 3] -> [B*wh*ww, WH*WW, 3, num_heads, dim_per_head] ->[qkv(3), B*wh*ww, nheads, WH*WW, dim_per_head]
        q, k, v = qkv.unbind(0)  # [B*wh*ww, nheads, WH*WW, dim_per_head]
        
        # scaled cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) # [B*wh*ww, nheads, WH*WW, WH*WW]，使用L2范数对q和k进行归一化(好处：尺度不影响度量；计算数值稳定性)，然后计算q和k的转置矩阵的乘积，得到注意力图
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()  # [num_heads, 1, 1]，对logit_scale进行截断，然后取指数，得到一个缩放因子
        attn = attn * logit_scale   # [B*wh*ww, nheads, WH*WW, WH*WW]
        attn = attn + self._get_rel_pos_bias()  # [B*wh*ww, nheads, WH*WW, WH*WW]

        checkerboard_attn_mask = Checkerboard_Window_Attention.checkerboard_attn_mask(win_size=self.window_size, device=x.device)  # [1, 1, WH*WW, WH*WW]

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(B_ // num_win, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            c1, c2, _, _ = attn.shape
            checkerboard_attn_mask = checkerboard_attn_mask.expand(c1, c2, -1, -1)
            attn.masked_fill_(~checkerboard_attn_mask, float('-inf')) 

            attn = self.softmax(attn)
            #注意softmax的作用是对每一行进行归一化，但是由于有些行因为mask的作用，全部为-inf，所以softmax以后为nan，需要将这些attn为nan的值变为0
            attn_mask = torch.isnan(attn)
            attn = attn.masked_fill(attn_mask, 0.0)

        else:
            c1, c2, _, _ = attn.shape
            checkerboard_attn_mask = checkerboard_attn_mask.expand(c1, c2, -1, -1)

            attn.masked_fill_(~checkerboard_attn_mask, float('-inf'))

            attn = self.softmax(attn)   # [B*wh*ww, nheads, WH*WW, WH*WW], 
            #注意softmax的作用是对每一行进行归一化，但是由于有些行因为mask的作用，全部为-inf，所以softmax以后为nan，需要将这些attn为nan的值变为0
            attn_mask = torch.isnan(attn)
            attn = attn.masked_fill(attn_mask, 0.0)
        
        attn = self.attn_drop(attn)     # [B*wh*ww, nheads, WH*WW, WH*WW]

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1) # [B*wh*ww, nheads, WH*WW, WH*WW] -> [B*wh*ww, nheads, WH*WW, dim_per_head] -> [B*wh*ww, WH*WW, nheads, dim_per_head] -> [B*wh*ww, WH*WW, nheads*dim_per_head]=[B*wh*ww, WH*WW, D]
        x = self.proj(x)        # [B*wh*ww, WH*WW, atten_d] -> [B*wh*ww, WH*WW, D]
        x = self.proj_drop(x)   # [B*wh*ww, WH*WW, D]
        return x

class Checkerboard_ESwinT(nn.Module): # [B, ph*pw, D], (ph, pw) 

    def __init__(
        self, dim, ngram, num_heads, window_size, shift_size,
        head_dim=None, mlp_ratio=4., qkv_bias=True, 
        drop=0., attn_drop=0., drop_path=0.,
        act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        
        super().__init__()
        self.dim = dim

        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        _assert(0 <= self.shift_size < self.window_size, "shift_size must in 0~window_size")
        
        self.ngram_window_partition = NWA(dim, window_size, ngram, num_heads, shift_size=shift_size)    # [B, ph, pw, D] -> [B*wh*ww, WH, WW, D], (wh, ww)
        self.attn = Checkerboard_Window_Attention(
            dim, num_heads=num_heads, head_dim=head_dim, window_size=to_2tuple(self.window_size),
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        
    def make_mask(self, num_patches):
        ph, pw = num_patches
        img_mask = torch.zeros((1, ph, pw, 1)) # [1, ph, pw, 1]
        cnt = 0
        for h in (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)):
            for w in (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows, (wh,ww) = window_partition(img_mask, self.window_size)  # [wh*ww*1, WH, WW, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # [wh*ww, WH*WW]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # [wh*ww, WH, WW]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0)) # [wh*ww, WH, WW]
        return attn_mask

    @staticmethod
    def checkerboard_window_mask(win_size, device):  # win_size=(WH,WW) win_mask=(1, 1, WH, WW) used for mask the attention map
        wh = win_size
        ww = win_size
        win_mask = np.ones((1, 1, wh, ww))
        win_mask[:, :, 0::2, 1::2] = 0
        win_mask[:, :, 1::2, 0::2] = 0
        win_mask = torch.tensor(win_mask, dtype=torch.float32, device=device)

        return win_mask

    def _attention(self, x, num_patches):   # [B, ph*pw, D], (ph, pw) -> [B, ph*pw, D], (ph, pw)
        # window partition - (cyclic shift) - cosine attention - window unpartition - (reverse shift)
        ph, pw = num_patches
        B, p, D = x.size()
        _assert(p == ph * pw, f"size is wrong!")
        
        # x = x.view(B, ph, pw, D) # [B, ph*pw, D] -> [B, ph, pw, D], Unembedding

        # x=[B, ph*pw, D] -> [B*wh*ww, D, WH, WW]
        x = Rearrange('b (num_wh wh num_ww ww) d -> (b num_wh num_ww) d wh ww', num_wh=ph//self.window_size, wh=self.window_size, ww=self.window_size)(x)

        c1, c2, _, _ = x.shape      # c1 = B*wh*ww, c2 = D

        checkboard_win_mask = Checkerboard_ESwinT.checkerboard_window_mask(win_size=self.window_size, device=x.device) # [1, 1, WH, WW]
        checkboard_win_mask = checkboard_win_mask.expand(c1, c2, -1, -1) # [B*wh*ww, D, WH, WW]

        x = x * checkboard_win_mask # [B*wh*ww, D, WH, WW]  # mask the input x

        # [B*wh*ww, D, WH, WW] -> [B, ph, pw, D]
        x = Rearrange('(b num_wh num_ww) d wh ww -> b (num_wh wh) (num_ww ww) d', num_wh=ph//self.window_size, num_ww=pw//self.window_size)(x)

        # N-Gram Window Partition (-> cyclic shift)
        x_windows, (wh,ww) = self.ngram_window_partition(x) # [B, ph, pw, D] -> [B*wh*ww, WH, WW, D], (wh, ww)
        
        x_windows = x_windows.view(-1, self.window_size * self.window_size, D)  # [B*wh*ww, WH*WW, D], Re-embedding
        
        # W-MSA/SW-MSA
        attn_mask = self.make_mask(num_patches).to(x.device) if self.shift_size>0 else None
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [B*wh*ww, WH*WW, D]
        
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, D) # [B*wh*ww, WH, WW, D], Unembedding
        
        # Window Unpartition
        shifted_x = window_unpartition(attn_windows, (wh,ww))  # [B, ph, pw, D]
        
        # Reverse Cyclic Shift
        reversed_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) if self.shift_size > 0 else shifted_x # [B, ph, pw, D]
        reversed_x = reversed_x.view(B, ph*pw, D) # [B, ph*pw, D], Re-embedding

        
        return reversed_x

    def forward(self, x, num_patches):
        x_ = x
        # (S)W Attention -> Layer-Norm -> Drop-Path -> Skip-Connection
        x = x + self.drop_path(self.norm1(self._attention(x, num_patches))) # [B, ph*pw, D]
        # FFN -> Layer-Norm -> Drop-Path -> Skip-Connection
        x = x + self.drop_path(self.norm2(self.ffn(x))) # [B, ph*pw, D]
        # return x_, x, num_patches
        return x, num_patches


class CB_ESTB(nn.Module):   # [B, C=320+i*slice, H, W] -> [B, C=320+i*slice, H, W]
    def __init__(self, in_dim, out_dim, ngram, window_size, num_heads, drop=0., attn_drop=0., drop_path=0., bottleneck_dim=192):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ngram = ngram
        self.window_size = window_size
        self.num_heads = num_heads
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path = drop_path
        self.bottleneck_dim = bottleneck_dim
        self.shift_size = [0, window_size // 2]

        self.cb_eswint1 = ESTB(dim=self.bottleneck_dim, 
                                             ngram=self.ngram, 
                                             num_heads=self.num_heads, 
                                             window_size=self.window_size, 
                                             shift_size=self.shift_size[0], 
                                             drop=self.drop, 
                                             attn_drop=self.attn_drop, 
                                             drop_path=self.drop_path)
        
        self.cb_eswint2 = ESTB(dim=self.bottleneck_dim,
                                                ngram=self.ngram,
                                                num_heads=self.num_heads,
                                                window_size=self.window_size,
                                                shift_size=self.shift_size[1],
                                                drop=self.drop,
                                                attn_drop=self.attn_drop,
                                                drop_path=self.drop_path)
        
        self.in_conv = conv1x1(self.in_dim, self.bottleneck_dim)
        self.out_conv = conv1x1(self.bottleneck_dim, self.out_dim)        
    
    def forward(self, x):
        # x: [B, C=320+i*slice, H, W] -> [B, C=320+i*slice, H, W]

        x = self.in_conv(x)  # [B, C=320+i*slice, H, W] -> [B, C=192, H, W]

        ph, pw = x.size(2), x.size(3)
        cb_eswint_in = Rearrange('b c h w -> b (h w) c')(x)   # [B, H*W, C=192]
        cb_eswint_out_1, patch_num = self.cb_eswint1(cb_eswint_in, (ph, pw))    # [B, H*W, C=192], (wh, ww)

        cb_eswint_out_2, patch_num = self.cb_eswint2(cb_eswint_out_1, patch_num)    # [B*wh*ww, WH*WW, D], (wh, ww)


        x_out = Rearrange('b (h w) c -> b c h w', h=patch_num[0])(cb_eswint_out_2)    # [B, C=192, H, W]

        x_out = self.out_conv(x_out)         # [B, C=192, H, W] -> [B, C=320+i*slice, H, W]

        return x_out


class ESTCB_SCC(nn.Module):
    '''
    if our paper is published, we will provide the details of the model.
    '''
  


if __name__ == "__main__":


    print("end")