import os
import sys
import math
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models
from pytorch_msssim import ms_ssim



class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, type='mse', epoislon=1e-7):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type
        self.epoislon = epoislon

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        epoislon_tensor = torch.tensor(self.epoislon, dtype=output["likelihoods"]["y"].dtype, device=output["likelihoods"]["y"].device)

        out["bpp_loss"] = sum(
            ((torch.log(likelihoods).sum() + epoislon_tensor) / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.type == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out['ms_ssim_loss'] = ms_ssim(output["x_hat"], target, data_range=1.)
            out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + out["bpp_loss"]

        return out

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

