import os
import json
import cv2
import shutil
import scipy
from math import log, exp
from scipy.interpolate import CubicSpline


from unittest.mock import patch

import numpy as np
import torch.nn.functional as F



def get_padding_size(height, width, p=128):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    # padding_left = (new_w - width) // 2
    padding_left = 0
    padding_right = new_w - width - padding_left
    # padding_top = (new_h - height) // 2
    padding_top = 0
    padding_bottom = new_h - height - padding_top
    return padding_left, padding_right, padding_top, padding_bottom


def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)


def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )


def Bjontegaard_Delta_Rate( 
# rate and psnr in ascending order 
    rate_ref, psnr_ref, # reference 
    rate_new, psnr_new, # new result 
):
    
    min_psnr = max(psnr_ref[0], psnr_new[0], 30)
    max_psnr = min(psnr_ref[-1], psnr_new[-1], 44) 
    log_rate_ref = [log(rate_ref[i]) for i in range(len(rate_ref))]
    log_rate_new = [log(rate_new[i]) for i in range(len(rate_new))]
    spline_ref = CubicSpline( 
        psnr_ref, log_rate_ref, bc_type='not-a-knot',
        extrapolate=True, 
    )

    spline_new = CubicSpline( 
        psnr_new, log_rate_new, bc_type='not-a-knot',
        extrapolate=True, 
    )
    
    delta_log_rate = (spline_new.integrate(min_psnr, max_psnr) - spline_ref.integrate(min_psnr, max_psnr))

    delta_rate = exp(delta_log_rate / (max_psnr - min_psnr))
    
    return 100 * (delta_rate - 1)

def evaluate_bd_rate_image(rate_ref, psnr_ref, rate_new, psnr_new):
    # def evaluate_bd_rate_image(image_dataset, ReferenceCodec, NewImageCodec):
    bd_rates = list() 
    # for image in image_dataset: 
    #     # evaluate rate and psnr on reference and new codec 
    #     # for this image with different qualities 
    #     rate_ref, psnr_ref = ReferenceCodec(image, qp=[...]) 
    #     rate_new, psnr_new = NewImageCodec(image, beta=[...]) 
    #     bd_rates.append( Bjontegaard_Delta_Rate( rate_ref, psnr_ref, rate_new, psnr_new, ) ) 
    #     # BD is computed per image and then averaged 
    
    # bd_rate = bd_rates.mean()

    bd_rates.append( Bjontegaard_Delta_Rate( rate_ref, psnr_ref, rate_new, psnr_new, ) ) 
    
    bd_rate = sum(bd_rates) / len(bd_rates)

    return bd_rate


def evaluate_bd_rate_video(video_dataset, ReferenceCodec, NewVideoCodec):
    bd_rates = list() 
    for video in video_dataset: 
        # evaluate rate and psnr on reference and new codec 
        # for this video with different qualities
        rate_ref, psnr_ref = ReferenceCodec(video, qp=[...]) 
        rate_new, psnr_new = NewVideoCodec(video, beta=[...]) 
        bd_rates.append( Bjontegaard_Delta_Rate( rate_ref, psnr_ref, rate_new, psnr_new, ) ) 
        # BD is computed per video and then averaged 
        
        bd_rate = bd_rates.mean()

        return bd_rate



def str2bool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1")


def interpolate_log(min_val, max_val, num, decending=True):
    assert max_val > min_val
    assert min_val > 0
    if decending:
        values = np.linspace(np.log(max_val), np.log(min_val), num)
    else:
        values = np.linspace(np.log(min_val), np.log(max_val), num)
    values = np.exp(values)
    return values


def scale_list_to_str(scales):
    s = ''
    for scale in scales:
        s += f'{scale:.2f} '

    return s


def generate_str(x):
    # print(x)
    if x.numel() == 1:
        return f'{x.item():.5f}  '
    s = ''
    for a in x:
        s += f'{a.item():.5f}  '
    return s


def create_folder(path, print_if_create=False):
    if not os.path.exists(path):
        os.makedirs(path)
        if print_if_create:
            print(f"created folder: {path}")


@patch('json.encoder.c_make_encoder', None)
def dump_json(obj, fid, float_digits=-1, **kwargs):
    of = json.encoder._make_iterencode  # pylint: disable=W0212

    def inner(*args, **kwargs):
        args = list(args)
        # fifth argument is float formater which we will replace
        args[4] = lambda o: format(o, '.%df' % float_digits)
        return of(*args, **kwargs)

    with patch('json.encoder._make_iterencode', wraps=inner):
        json.dump(obj, fid, **kwargs)


def check_and_copy(img_dir:str, save_dir:str, img_height:int=384, img_width:int=384):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_list = os.listdir(img_dir)
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img.shape[0] >= img_height and img.shape[1] >= img_width:
            shutil.copy(img_path, save_dir)

    print("end")


def ImageNet_check_and_copy(Image_net_train_dir:str, dest_dir:str, img_height:int, img_width:int):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    else:
        os.system(f"rm -rf {dest_dir}/*")

    img_subdir_list = os.listdir(Image_net_train_dir)
    for img_subdir in img_subdir_list:
        if os.path.isdir(os.path.join(Image_net_train_dir, img_subdir)):
            img_subdir_path = os.path.join(Image_net_train_dir, img_subdir)
            img_list = os.listdir(img_subdir_path)
            for img_name in img_list:
                img_path = os.path.join(img_subdir_path, img_name)
                img = cv2.imread(img_path)
                if img.shape[0] >= img_height and img.shape[1] >= img_width:
                    shutil.copy(img_path, dest_dir)

    img_list = os.listdir(dest_dir)
    print(f"ImageNet dataset checked has {len(img_list)} images")


    print("end")





if __name__ == "__main__":

    rate_ref, psnr_ref = ((0.278, 0.427, 0.629, 0.9), (31.55, 33.36, 35.13, 36.83)) 
    rate_new, psnr_new = ((0.282, 0.426, 0.621, 0.92), (31.88, 33.68, 35.39, 36.94))

    print(f"rate_ref: {rate_ref}") 
    print(f"psnr_ref: {psnr_ref}")
    


    bd_rate = evaluate_bd_rate_image(rate_ref=rate_ref, psnr_ref=psnr_ref, rate_new=rate_new, psnr_new=psnr_new)

    print(f"bd_rate: {bd_rate}")


    print("end")