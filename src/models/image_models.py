import os
import math
import cv2
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


from layers import Bicubic, Encoder
from entropy_models import BitEstimator
# from src.utils.commond import get_padding_size



def base_function(x, a=-0.5):
    # describe the base function sin(x)/x
    Wx = 0
    if np.abs(x)<=1:
        Wx = (a+2)*(np.abs(x)**3) - (a+3)*x**2 + 1
    elif 1<=np.abs(x)<=2:
        Wx = a*(np.abs(x)**3) - 5*a*(np.abs(x)**2) + 8*a*np.abs(x) - 4*a
    return Wx

def padding(img):
    h, w, c = img.shape
    pad_image = np.zeros((h+4, w+4, c))
    pad_image[2:h+2, 2:w+2] = img
    return pad_image

def draw_function():
    a = -0.5
    x = np.linspace(-3.0, 3.0, 100)
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = base_function(x[i], a)
    plt.figure("base_function")
    plt.plot(x, y)
    plt.show()

def bicubic(img, sacle, a=-0.5):
    h, w, color = img.shape
    img = padding(img)
    nh = h*sacle
    nw = h*sacle
    new_img = np.zeros((nh, nw, color))

    for c in range(color):
        for i in range(nw):
            for j in range(nh):

                px = i/sacle + 2
                py = j/sacle + 2
                px_int = int(px)
                py_int = int(py)
                u = px - px_int
                v = py - py_int

                A = np.matrix([[base_function(u+1, a)], [base_function(u, a)], [base_function(u-1, a)], [base_function(u-2, a)]])
                C = np.matrix([base_function(v+1, a), base_function(v, a), base_function(v-1, a), base_function(v-2, a)])
                B = np.matrix([[img[py_int-1, px_int-1][c], img[py_int-1, px_int][c], img[py_int-1, px_int+1][c], img[py_int-1, px_int+2][c]],
                               [img[py_int, px_int-1][c], img[py_int, px_int][c], img[py_int, px_int+1][c], img[py_int, px_int+2][c]],
                               [img[py_int+1, px_int-1][c], img[py_int+1, px_int][c], img[py_int+1, px_int+1][c], img[py_int+1, px_int+2][c]],
                               [img[py_int+2, px_int-1][c], img[py_int+2, px_int][c], img[py_int+2, px_int+1][c], img[py_int+2, px_int+2][c]]])
                new_img[j, i][c] = np.dot(np.dot(C, B), A)
    return new_img



class ISRC_Net(nn.Module):
    def __init__(self, scaling_model:str, scaling_factor:int, input_channel:int, y_channel:int, z_channel:int):
        super().__init__()

        self.scaling_factor = scaling_factor
        self.y_channel = y_channel
        self.z_channel = z_channel

        if scaling_model == "bicubic":
            self.x_scale = Bicubic(scale=1.0 / scaling_factor)
        else:
            self.x_scale = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=scaling_factor)       # Degradation: downsmapling

        
        self.encoder = Encoder(input_channel=input_channel, channel=y_channel)


        # self.bit_estimator_y = 
        self.bit_estimator_z = BitEstimator(z_channel)



    @staticmethod
    def bicubic(img, sacle):
        return F.interpolate(img, scale_factor=sacle, mode="bicubic", align_corners=False)

    @staticmethod
    def probs_to_bits(probs):
        bits = -1.0 * torch.log(probs + 1e-5) / math.log(2.0)
        bits = torch.sum(torch.clamp(bits, 0, 50))
        return bits
    
    # get the bits of y and z by entropy model, rather than the actual bits
    def get_y_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10) # lowbound is same as to bits calculation
        gaussian_distri = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian_distri.cdf(y + 0.5) - gaussian_distri.cdf(y - 0.5)
        bits = ISRC_Net.probs_to_bits(probs)
        return bits

    def get_z_bits(self, z):
        probs = self.bit_estimator_z(z + 0.5) - self.bit_estimator_z(z - 0.5)
        bits = ISRC_Net.probs_to_bits(probs)
        return bits



    def forward(self, x):
        x_down = self.x_scale(x)
        y = self.encoder(x_down)


        return x_down, y
    



if __name__ == "__main__":

    img_1 = cv2.imread("E:\Desktop_Daily\Paper_and_Coding\\01_image_compression\z_my_work\dataset_test\\vimeo_im1.png", flags=1)
    img = torch.from_numpy(img_1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    x = img
    model_bicubic = ISRC_Net(scaling_model="bicubic", scaling_factor=2, input_channel=3, y_channel=192, z_channel=192)
    x_down_bicubic, y_bicubic = model_bicubic(x)

    model_cov = ISRC_Net(scaling_model="conv_down", scaling_factor=2, input_channel=3, y_channel=192, z_channel=192)
    x_down_cov, y_conv = model_cov(x)

    # img_conv = x_conv.squeeze(0).permute(1, 2, 0).detach().numpy() * 255.0
    # img_conv = img_conv.clip(0, 255).astype("uint8")

    # plt.subplot(1, 3, 1)
    # plt.imshow(img_1)
    # plt.subplot(1, 3, 2)
    # plt.imshow(img_conv)
    # plt.subplot(1, 3, 3)
    # plt.imshow(img_bicubic)
    # plt.show()

    print(f"x_origin: {x.shape}")

    print(f"x_bicubic: {x_down_bicubic.shape}")
    print(f"x_conv: {x_down_cov.shape}")

    print(f"y_bicubic: {y_bicubic.shape}")
    print(f"y_conv: {y_conv.shape}")

    print("end")
