import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import scipy
from math import log, exp
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.commond import get_padding_size


# pylint: disable=W0221
class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None
# pylint: enable=W0221


def conv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv3x3(in_ch, out_ch, r=1):
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )


def subpel_conv1x1(in_ch, out_ch, r=1):
    """1x1 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r)
    )


def conv1x1(in_ch, out_ch, stride=1, bias=True):
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, bias=bias, kernel_size=1, stride=stride)




# 有问题，需要重新写
class GDN(nn.Module):
    """Generalized divisive normalization layer.
    Args:
        ch (int): number of channels
        inverse (bool): whether to compute inverse GDN
        beta_min (float): lower bound for beta
    """

    def __init__(self, ch, inverse=False, beta_min=1e-6):
        super().__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gain = nn.Parameter(torch.ones(ch))
        self.bias = nn.Parameter(torch.zeros(ch))
        self.beta = nn.Parameter(torch.ones(ch) * 2)

    def forward(self, x):
        if self.inverse:
            return self._inverse(x)
        else:
            return self._forward(x)

    def _forward(self, x):
        gain = self.gain.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        norm_pool = F.avg_pool2d(x ** 2, (x.size(2), x.size(3)), stride=1, padding=0)
        x = x * torch.rsqrt(norm_pool * beta ** 2 + self.beta_min)
        x = gain * x + bias
        return x

    def _inverse(self, x):
        gain = self.gain.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        norm_pool = F.avg_pool2d(x ** 2, (x.size(2), x.size(3)), stride=1, padding=0)
        gain = 1.0 / gain
        beta = beta / (norm_pool * beta ** 2 + self.beta_min)
        x = (x - bias) * gain * torch.sqrt(beta)
        return x


class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.
    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """
    def __init__(self, in_ch1, in_ch2, in_ch3=None, stride=2):
        super().__init__()
        if in_ch3 is None:
            in_ch3 = in_ch2
        self.conv1 = conv3x3(in_ch1, in_ch2, stride=stride)
        self.leaky_relu = nn.LeakyReLU()
        self.conv2 = conv3x3(in_ch2, in_ch3)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        if stride != 1:
            self.downsample = conv1x1(in_ch1, in_ch3, stride=stride)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return out


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch1, in_ch2, in_ch3=None, upsample=2):
        super().__init__()
        if in_ch3 is None:
            in_ch3 = in_ch2
        self.subpel_conv = subpel_conv1x1(in_ch1, in_ch2, upsample)
        self.leaky_relu = nn.LeakyReLU()
        self.conv = conv3x3(in_ch2, in_ch3)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.upsample = subpel_conv1x1(in_ch1, in_ch3, upsample)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.leaky_relu2(out)
        identity = self.upsample(x)
        out = out + identity
        return out


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch1, in_ch2, in_ch3=None, leaky_relu_slope=0.01):
        super().__init__()
        if in_ch3 is None:
            in_ch3 = in_ch2
        self.conv1 = conv3x3(in_ch1, in_ch2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.conv2 = conv3x3(in_ch2, in_ch3)
        self.adaptor = None
        if in_ch1 != in_ch3:
            self.adaptor = conv1x1(in_ch1, in_ch3)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        out = out + identity
        return out


class DepthConv(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3=None, depth_kernel=3, stride=1):
        super().__init__()
        if in_ch3 is None:
            in_ch3 = in_ch2
            in_ch2 = in_ch1
        # dw_ch = in_ch * 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch1, in_ch2, 1, stride=stride),
            nn.LeakyReLU(),
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_ch2, in_ch2, depth_kernel, padding=depth_kernel // 2, groups=in_ch2),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch2, in_ch3, 1),
            nn.LeakyReLU(),
        )
        self.adaptor = None
        if stride != 1:
            assert stride == 2
            self.adaptor = nn.Conv2d(in_ch1, in_ch3, 2, stride=2)
        elif in_ch1 != in_ch3:
            self.adaptor = nn.Conv2d(in_ch1, in_ch3, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.depth_conv(out)
        out = self.conv2(out)

        return out + identity


class ConvFFN(nn.Module):
    def __init__(self, in_ch1, in_ch2=None):
        super().__init__()
        if in_ch2 is None:
            in_ch2 = in_ch1 * 4
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch1, in_ch2, 1),
            nn.ReLU(),
            nn.Conv2d(in_ch2, in_ch1, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        identity = x
        return identity + self.conv(x)


class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, out_ch, depth_kernel=depth_kernel, stride=stride),
            ConvFFN(out_ch),
        )

    def forward(self, x):
        return self.block(x)

class Bicubic(nn.Module):
    def __init__(self, scale):
        super().__init__()

        self.scale = scale

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)


def get_enc_dec_models(input_channel, output_channel, channel):
    enc = nn.Sequential(
        ResidualBlockWithStride(input_channel, channel, stride=2),
        ResidualBlock(channel, channel),
        ResidualBlockWithStride(channel, channel, stride=2),
        ResidualBlock(channel, channel),
        ResidualBlockWithStride(channel, channel, stride=2),
        ResidualBlock(channel, channel),
        conv3x3(channel, channel, stride=2),
    )

    dec = nn.Sequential(
        ResidualBlock(channel, channel),
        ResidualBlockUpsample(channel, channel, 2),
        ResidualBlock(channel, channel),
        ResidualBlockUpsample(channel, channel, 2),
        ResidualBlock(channel, channel),
        ResidualBlockUpsample(channel, channel, 2),
        ResidualBlock(channel, channel),
        subpel_conv1x1(channel, output_channel, 2),
    )

    return enc, dec


def get_hyper_enc_dec_models(y_channel, z_channel):
    enc = nn.Sequential(
        conv3x3(y_channel, z_channel),
        nn.LeakyReLU(),
        conv3x3(z_channel, z_channel),
        nn.LeakyReLU(),
        conv3x3(z_channel, z_channel, stride=2),
        nn.LeakyReLU(),
        conv3x3(z_channel, z_channel),
        nn.LeakyReLU(),
        conv3x3(z_channel, z_channel, stride=2),
    )

    dec = nn.Sequential(
        conv3x3(z_channel, y_channel),
        nn.LeakyReLU(),
        subpel_conv1x1(y_channel, y_channel, 2),
        nn.LeakyReLU(),
        conv3x3(y_channel, y_channel * 3 // 2),
        nn.LeakyReLU(),
        subpel_conv1x1(y_channel * 3 // 2, y_channel * 3 // 2, 2),
        nn.LeakyReLU(),
        conv3x3(y_channel * 3 // 2, y_channel * 2),
    )

    return enc, dec


class Encoder(nn.Module):
    def __init__(self, input_channel, channel):
        super().__init__()
        self.enc = nn.Sequential(
            ResidualBlockWithStride(input_channel, channel, stride=2),
            ResidualBlock(channel, channel),
            ResidualBlockWithStride(channel, channel, stride=2),
            ResidualBlock(channel, channel),
            ResidualBlockWithStride(channel, channel, stride=2),
            ResidualBlock(channel, channel),
            conv3x3(channel, channel, stride=2),
        )

    def forward(self, x):
        return self.enc(x)
    

class Hyper_Encoder(nn.Module):
    def __init__(self, y_channel, z_channel):
        super().__init__()
        self.enc = nn.Sequential(
            conv3x3(y_channel, z_channel),
            nn.LeakyReLU(),
            conv3x3(z_channel, z_channel),
            nn.LeakyReLU(),
            conv3x3(z_channel, z_channel, stride=2),
            nn.LeakyReLU(),
            conv3x3(z_channel, z_channel),
            nn.LeakyReLU(),
            conv3x3(z_channel, z_channel, stride=2),
        )

    def forward(self, x):
        return self.enc(x)
    

class Hyper_Decoder(nn.Module):
    def __init__(self, y_channel, z_channel):
        super().__init__()
        self.dec = nn.Sequential(
            conv3x3(z_channel, y_channel),
            nn.LeakyReLU(),
            subpel_conv1x1(y_channel, y_channel, 2),
            nn.LeakyReLU(),
            conv3x3(y_channel, y_channel * 3 // 2),
            nn.LeakyReLU(),
            subpel_conv1x1(y_channel * 3 // 2, y_channel * 3 // 2, 2),
            nn.LeakyReLU(),
            conv3x3(y_channel * 3 // 2, y_channel * 2),
        )

    def forward(self, x):
        return self.dec(x)








if __name__ == "__main__":


    img_original = torch.randn(1, 3, 256, 448)
    pic_height, pic_width = img_original.shape[2], img_original.shape[3]

    # pad if necessary
    padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width)

    x_padded = F.pad(img_original,
        (padding_l, padding_r, padding_t, padding_b),
        mode="constant",
        value=0,)
    
    # x_hat = F.pad(recon_frame, (-padding_l, -padding_r, -padding_t, -padding_b))    
    


    img = Bicubic(1/2)(x_padded)


    enc = Encoder(3, 192)
    hyper_encoder = Hyper_Encoder(192, 192)
    hyper_decoder = Hyper_Decoder(192, 192)

    y = enc(img)
    z = hyper_encoder(y)
    z_hat = hyper_decoder(z)

    print(f"img_original: {img_original.shape}")
    print(f"x_padded: {x_padded.shape}")

    print(f"img_down: {img.shape}")
    print(f"y: {y.shape}")
    print(f"z: {z.shape}")
    print(f"z_hat: {z_hat.shape}")


    print("end")




