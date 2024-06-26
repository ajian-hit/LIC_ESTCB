import numpy as np
import pandas as pd
import torch
import math
import os

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
import torch.nn.functional as F
import torchvision
from PIL import Image
import cv2


from pandas import DataFrame as df
from sklearn.datasets import load_wine, load_breast_cancer
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn.metrics as skm
# import ssl


# import test model path
import sys
sys.path.append(r"E:\Desktop_Daily\\Paper_and_Coding\\01_image_compression\\z_my_work")

from model.archs.SCEST_no_AE_v2 import SCESwinT_SCC_new

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

def heatmap_full_params():

    # step1: 加载数据
    # method1: 通过pandas直接读取csv文件，返回DataFrame类型
    # method2: 通过load_dataset()函数，返回DataFrame类型
    data = pd.read_csv(r"C:\\Users\DJ Yang\seaborn-data\\flights.csv", encoding="utf-8", engine="python")

    # step2: 数据预处理，排序，转换数据类型等，返回DataFrame类型
    # 1. 将某些不可以按照默认排序的非有序分类变量转换为有序分类变量
    month = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October','November', 'December']
    data['month'] = pd.Categorical(data['month'], categories=month, ordered=True) # 将月份转换为有序分类变量
    # 2. 其余可以按照默认排序的分类变量转换为有序分类变量
    # data = data.sort_values(by=['year'], ascending=True) # 按照年份排序(升序打开ascending=True)

    # step3: 重塑数据，返回DataFrame类型
    data = data.pivot(index='month', columns='year', values='passengers') # 重塑数据，行为月份，列为年份，值为乘客数

    # step4: 显示数据，返回None
    print(data.head(12))    # 打印数据，其中12表示打印前12行，如果不写，默认为5行
    data.head().info()      # 打印数据的信息


    # step5: 绘制热力图，返回None
    # 1. 设置字体和画布大小
    sns.set(font_scale=1) # 设置字体大小
    sns.set_context(rc={'figure.figsize':(32, 32)}) # 设置画布大小
    # plt.figure(figsize=(8, 6))
    # 2. 绘制热力图
    # 参数介绍：https://blog.csdn.net/m0_38103546/article/details/79935671
    sns.heatmap(data=data,   # 设置数据，原始数据，相关性数据（data=data.corr()）
                annot=True, annot_kws={'size':5,'weight':'bold', 'color':'red'}, # 显示每个方格的数据,并设置格式
                fmt='.1f',   # 设置数据格式
                cmap='RdBu_r', # 设置颜色带的色系,[色彩选择https://blog.csdn.net/ztf312/article/details/102474190,_r是反转颜色]
                # center=300,  # 设置颜色带的分界线
                # vmin=200,    # 设置颜色带的最小值
                # vmax=500,    # 设置颜色带的最大值

                # xticklabels=month, # 设置x轴标签
                # yticklabels=month, # 设置y轴标签
                linewidths=0.05, linecolor="white" # 设置方格间隔
                )
    # 3. 显示、保存图片
    # plt.show()  # 显示图片
    plt.savefig(r"E:\Desktop_Daily\\Paper_and_Coding\\01_image_compression\\z_my_work\\heatmap.png", dpi=400, bbox_inches='tight') # bbox_inches='tight'表示保存图片时将图片四周的空白区域全部剪掉
    

def heatmap(data, cmap, save_path, v_index=False, vmin=None, vmax=None, annot=False):

    # step5: 绘制热力图，返回None
    # 1. 设置字体和画布大小
    sns.set(font_scale=1) # 设置字体大小
    sns.set_context(rc={'figure.figsize':(32, 32)}) # 设置画布大小
    # plt.figure(figsize=(8, 6))
    # 2. 绘制热力图
    # 参数介绍：https://blog.csdn.net/m0_38103546/article/details/79935671

    if not v_index:
        sns.heatmap(data=data,   # 设置数据，原始数据，相关性数据（data=data.corr()）
                    annot=annot, annot_kws={'size':10,'weight':'bold', 'color':'red'}, # 显示每个方格的数据,并设置格式
                    fmt='.4f',      # 设置数据格式
                    cmap=cmap,      # 设置颜色带的色系,[色彩选择https://blog.csdn.net/ztf312/article/details/102474190,_r是反转颜色]
                    # center=300,   # 设置颜色带的分界线
                    # vmin=vmin,      # 设置颜色带的最小值
                    # vmax=vmax,      # 设置颜色带的最大值

                    xticklabels=False, # 设置x轴标签
                    yticklabels=False, # 设置y轴标签

                    # linewidths=0.05 # 设置方格间隔
                    )
    else:
        sns.heatmap(data=data,   # 设置数据，原始数据，相关性数据（data=data.corr()）
                    annot=annot, annot_kws={'size':10,'weight':'bold', 'color':'red'}, # 显示每个方格的数据,并设置格式
                    fmt='.4f',      # 设置数据格式
                    cmap=cmap,      # 设置颜色带的色系,[色彩选择https://blog.csdn.net/ztf312/article/details/102474190,_r是反转颜色]
                    # center=300,   # 设置颜色带的分界线
                    vmin=vmin,      # 设置颜色带的最小值
                    vmax=vmax,      # 设置颜色带的最大值

                    xticklabels=False, # 设置x轴标签
                    yticklabels=False, # 设置y轴标签

                    # linewidths=0.05 # 设置方格间隔
                    )
        

    # 3. 显示、保存图片
    # plt.show()  # 显示图片
    plt.savefig(save_path, 
                dpi=400,             # dpi：
                bbox_inches='tight', # bbox_inches='tight'表示保存图片时将图片四周的空白区域全部剪掉
                )
    print(f"heatmap is saved in {save_path}")


def pairplot():
    data = df(load_breast_cancer().data, columns=load_breast_cancer().feature_names)
    data['target'] =  load_breast_cancer().target

    # print(data.head())
    # data.head().info()


    data1 = data.iloc[:,-5:]
    print(data1["target"].value_counts())
    print(data1.head())
    data1.head().info()
    sns.pairplot(data1)
    plt.show()


    print("end")


def y_latent_draw_spatial_correlation(x_vector, y_vector):
    '''
    返回的向量的【行数，列数】= 【矩阵1的行数 + 矩阵2的行数】
    '''
    assert x_vector.shape == y_vector.shape, print("x_vector.shape must equal to y_vector.shape")
    assert len(x_vector.shape) == 1, print("x_vector.shape[0] must equal to 1")

    corr_coef = abs(np.corrcoef(x_vector, y_vector)[0, 1])   # 计算相关系数

    return corr_coef


def y_latent_normalized_avarage_in_Kodak(result_path):

    # step1: 遍历文件夹
    file_list = os.listdir(result_path)
    file_name_suffix_list = []
    for file_name in file_list:
        if "y_latent" in file_name:
            file_name_suffix_list.append(file_name.split("_")[2] + "_" + file_name.split("_")[3])

    latent_y_list = []
    means_hat_list = []
    scales_hat_list = []

    latent_name = "y_latent"
    means_name = "means_hat"
    scales_name = "scales_hat"

    for file_name_suffix in file_name_suffix_list:
        latent_y_path = os.path.join(result_path, latent_name + "_" + file_name_suffix)
        means_hat_path = os.path.join(result_path, means_name + "_" + file_name_suffix)
        scales_hat_path = os.path.join(result_path, scales_name + "_" + file_name_suffix)

        latent_y = torch.load(latent_y_path)
        means_hat = torch.load(means_hat_path)
        scales_hat = torch.load(scales_hat_path)

        latent_y_list.append(latent_y)
        means_hat_list.append(means_hat)
        scales_hat_list.append(scales_hat)



    # step2: 计算每一张图片的y_latent的平均值
    latent_y_numpy_list = []
    means_hat_numpy_list = []
    scales_hat_numpy_list = []

    latent_y_normalized_avarage_list = []

    for i in range(len(latent_y_list)):
        latent_y_numpy_list.append(latent_y_list[i].data.cpu().numpy())
        means_hat_numpy_list.append(means_hat_list[i].data.cpu().numpy())
        scales_hat_numpy_list.append(scales_hat_list[i].data.cpu().numpy())

        latent_y_normalized_avarage_list.append((np.round(latent_y_numpy_list[i]) - means_hat_numpy_list[i]) * 1.0 / scales_hat_numpy_list[i])

        # latent_y_normalized_avarage_list.append((latent_y_numpy_list[i] - means_hat_numpy_list[i] + 1e-10) * 1.0 / scales_hat_numpy_list[i] + 1e-10)

    
    print(f"len(latent_y_normalized_avarage_list): {len(latent_y_normalized_avarage_list)}")
    print(f"len(latent_y_normalized_avarage_list[0]): {latent_y_normalized_avarage_list[0].shape}")
    print(f"return value is a list, include 24 numpy array of (1, 192, H, W), each numpy array is a image's y_latent_normalized_avarage")

    # print(f"latent_y_normalized_avarage_list: {latent_y_normalized_avarage_list}")

    return latent_y_normalized_avarage_list


def y_latent_entropy_claculate_and_highest_entropy_layer_index(y_latent, means_params, sigma_params, prob_model:str="Gaussian"):

    assert prob_model in ["Gaussian"], print("prob_model must Gaussian, if not, need to modify the prob_model")

    # sigma = sigma_params
    # mu = torch.zeros_like(sigma)
    # sigma = sigma.clamp(1e-5, 1e10) # lowbound is same as to bits calculation

    sigma = sigma_params.clamp(1e-5, 1e10) # lowbound is same as to bits calculation
    mu = means_params.clamp(1e-5, 1e10) # lowbound is same as to bits calculation

    gaussian_distri = torch.distributions.normal.Normal(mu, sigma)
    probs = gaussian_distri.cdf(y_latent + 0.5) - gaussian_distri.cdf(y_latent - 0.5)

    bits = -1.0 * torch.log(probs + 1e-5) / math.log(2.0)
    bits = [bits[:, i, :, :] for i in range(bits.shape[1])]
    layer_bits_sum = [torch.sum(bits[i]).item() for i in range(len(bits))]  # 每一层的bits, 即entropy
    max_entropy_layer_index, max_entropy_value = max(enumerate(layer_bits_sum), key=lambda x: x[1]) # 最大entropy的层的index及其值

    return layer_bits_sum, max_entropy_layer_index, max_entropy_value 
    

def draw_y_latent_highest_entropy_layer_yquan_means_sigma():

    y_latent_path = "E:\Desktop_Daily\Paper_and_Coding\\01_image_compression\z_my_work\z_linshi_2\means_hat.pt"
    means_hat_path = "E:\Desktop_Daily\Paper_and_Coding\\01_image_compression\z_my_work\z_linshi_2\means_hat.pt"
    scales_hat_path = "E:\Desktop_Daily\Paper_and_Coding\\01_image_compression\z_my_work\z_linshi_2\scales_hat.pt"

    y_latent = torch.load(y_latent_path)
    means_hat = torch.load(means_hat_path)
    scales_hat = torch.load(scales_hat_path)

    print(f"y_latent.shape: {y_latent.shape}")
    print(f"means_hat.shape: {means_hat.shape}")
    print(f"scales_hat.shape: {scales_hat.shape}")

    y_latent_highest_entropy_layer = np.round(y_latent[0, 15, :, :].data.cpu().numpy())
    means_hat_highest_entropy_layer = means_hat[0, 15, :, :].data.cpu().numpy()
    scales_hat_highest_entropy_layer = scales_hat[0, 15, :, :].data.cpu().numpy()
    re_redundundancy = (y_latent_highest_entropy_layer - means_hat_highest_entropy_layer) * 1.0 / scales_hat_highest_entropy_layer
    print(f"re_redundundancy.shape: {re_redundundancy.shape}")
    print(f"re_redundundancy.max(): {re_redundundancy.max()}")

    plt.subplot(1, 4, 1)
    heatmap(y_latent_highest_entropy_layer,
            cmap="RdBu_r",
            v_index=False, 
            save_path="E:\Desktop_Daily\Paper_and_Coding\\01_image_compression\z_my_work\z_linshi_2\\y_latent_highest_entropy_layer.png")
    plt.subplot(1, 4, 2)
    heatmap(means_hat_highest_entropy_layer, 
            cmap="RdBu_r",
            v_index=False,
            save_path="E:\Desktop_Daily\Paper_and_Coding\\01_image_compression\z_my_work\z_linshi_2\\means_hat_highest_entropy_layer.png")
    plt.subplot(1, 4, 3)
    heatmap(scales_hat_highest_entropy_layer, 
            cmap="Blues",
            v_index=True,
            vmax=40,
            vmin=0,
            save_path="E:\Desktop_Daily\Paper_and_Coding\\01_image_compression\z_my_work\z_linshi_2\\scales_hat_highest_entropy_layer.png")
    plt.subplot(1, 4, 4)
    heatmap(re_redundundancy, 
            cmap="RdBu_r",
            v_index=True,
            vmax=1,
            vmin=-1,
            save_path="E:\Desktop_Daily\Paper_and_Coding\\01_image_compression\z_my_work\z_linshi_2\\re_redundundancy.png")
    plt.show()


    print("end")


def draw_spatial_corr_of_y_latent(y_latent_normalized_avarage_in_Kodak, x_base, y_base, x_bias, y_bias):

    corr_matrix_of_y_latent_list = []

    for index in range(len(y_latent_normalized_avarage_in_Kodak)):
        y_latent_base_vector = y_latent_normalized_avarage_in_Kodak[index][0, :, x_base, y_base]
        print(f"y_latent_base_vector_list[{index}].shape: {y_latent_base_vector.shape}")
        x_index, y_index = [2 * x_bias + 1, 2 * y_bias + 1]
        corr_matrix_of_y_latent = np.empty(shape=[x_index, y_index])

        for i in range(x_index):
            for j in range(y_index):
                y_latent_i_j_vector = y_latent_normalized_avarage_in_Kodak[index][0, :, x_base-x_bias+i, y_base-y_bias+j].reshape(-1)
                corr_matrix_of_y_latent[i, j] = y_latent_draw_spatial_correlation(y_latent_base_vector, y_latent_i_j_vector)

        corr_matrix_of_y_latent_list.append(corr_matrix_of_y_latent)
    
    print(f"len(corr_matrix_of_y_latent_list): {len(corr_matrix_of_y_latent_list)}")
    print(f"corr_matrix_of_y_latent_list[0].shape: {corr_matrix_of_y_latent_list[0].shape}")

    # get average
    corr_matrix_of_y_latent_avarage = np.zeros(shape=corr_matrix_of_y_latent_list[0].shape)
    for i in range(len(corr_matrix_of_y_latent_list)):
        corr_matrix_of_y_latent_avarage += corr_matrix_of_y_latent_list[i]
    corr_matrix_of_y_latent_avarage /= len(corr_matrix_of_y_latent_list)

    print(f"corr_matrix_of_y_latent_avarage.shape: {corr_matrix_of_y_latent_avarage.shape}")


    # draw heatmap
    heatmap(corr_matrix_of_y_latent_avarage, 
            cmap="Blues",
            annot=True,
            v_index=True,
            vmax=0.5,
            vmin=0,
            save_path="E:\Desktop_Daily\Paper_and_Coding\\01_image_compression\z_my_work\z_linshi_2\\corr_matrix_of_y_latent_average.png")
    
    plt.show()

    print("end")


def draw_ERF(model, path, pre_checkpoint, base_check_pos, delta_x, delta_y, min_v, max_v, save_path):
    '''
    params:
    model: the model of the network
    path: the path of the image
    pre_checkpoint: the path of the pre-trained model
    base_check_pos: the base point of the heatmap
    delta_x: the x_bias from the base point of the heatmap
    delta_y: the y_bias from the base point of the heatmap
    min_v: the min value of the heatmap
    max_v: the max value of the heatmap
    save_path: the path of the saved image


    注意事项：
    1、首先选定一个基准点base_check_pos，然后取一个范围【Δx,Δy】，遍历热力图，看哪个更符合自己想要的；
    2、热力图与原图的合并可以调整透明度的权重，这样可以更好的看到原图的细节；
    3、热力图的颜色可以调整，这样可以更好的看到原图的细节；
    '''
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 选择设备
    net = model.to(device)                                                  # 在设备上加载模型
    net.eval()                                                              # 设置为评估模式

    dictory = {}
    print(f"Loading {pre_checkpoint}")
    checkpoint = torch.load(pre_checkpoint, map_location=device)
    for k, v in checkpoint["state_dict"].items():
        dictory[k.replace("module.", "")] = v
    net.load_state_dict(dictory)
    print(f"Loading {pre_checkpoint} success")                              # 加载预训练模型


    p = 128                                                                 # 设置p的值, 用于对图片进行padding
    img = transforms.ToTensor()(Image.open(path).convert('RGB')).to(device) # 读取图片
    x = img.unsqueeze(0)
    x_padded, padding = pad(x, p)                                           # 对图片进行padding

    x_padded.requires_grad = True                                           # 设置x_padded的梯度为True

    x_hat = net.forward(x_padded)["x_hat"]                                  # 前向传播

    x_hat_fmap = x_hat.mean(dim=1,keepdim=False).squeeze()                  # 取出x_hat的feature map(可以修改，选择自己想要的feature map)

    print(f"the shape of x_hat_fmap is {x_hat_fmap.shape}")                 # 打印x_hat_fmap的shape


    # the backward point of x_hat_fmap is (base_check_pos[0] + delta_x, base_check_pos[1] + delta_y)

    x_hat_fmap[base_check_pos[0]+delta_x][base_check_pos[1]+delta_y].backward()  # 反向传播


    grad = torch.abs(x_padded.grad)                                         # 取出梯度的绝对值
    grad = grad.mean(dim=1,keepdim=False).squeeze()                         # 取出梯度的平均值（按照channel维度）

    heatmap = grad.cpu().numpy()                                            # 将梯度转换为numpy格式
    print(f"the shape of heatmap is {heatmap.shape}")
    print(f"the max of heatmap is {np.max(heatmap)}")
    print(f"the min of heatmap is {np.min(heatmap)}")

    heatmap = np.clip(heatmap, min_v, max_v)                                # 将heatmap的值限制在[min_v, max_v]之间

    heatmap = heatmap * 1.0 / np.max(heatmap)                               # 将heatmap的值归一化



    cam = cv2.applyColorMap(np.uint8(heatmap*255), cv2.COLORMAP_JET)        # 将heatmap转换为热力图
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)                              # 将热力图转换为RGB格式
    Image.fromarray(cam)                                                    # 将热力图转换为PIL格式
    # plt.imshow(cam)

    img_original = Image.open(path)                                         # 读取原图

    add_img = cv2.addWeighted(np.array(img_original), 0.6, cam, 0.4, 0)     # 将原图与热力图进行融合

    plt.imshow(add_img)                                                     # 显示融合后的图片
    plt.show()                                                              # 显示图片

    add_img.save(save_path)                                                 # 保存图片


    print("end")


def work():
    img_dir = "E:\Desktop_Daily\Paper_and_Coding\\01_image_compression\z_my_work\dataset_test\\voc_100\save\JPEGImages"
    images=os.listdir(img_dir)
    model = deeplabv3_resnet50(pretrained=True, progress=False)
    model = model.eval()
    #定义输入图像的长宽，这里需要保证每张图像都要相同
    input_H, input_W = 512, 512
    #生成一个和输入图像大小相同的0矩阵，用于更新梯度
    heatmap = np.zeros([input_H, input_W])
    #打印一下模型，选择其中的一个层
    print(model)

    #这里选择骨干网络的最后一个模块
    layer = model.backbone.layer4[-1]
    print(layer)


    def farward_hook(module, data_input, data_output):
        fmap_block.append(data_output)
        input_block.append(data_input)
        
    # 为了简单，这里直接一张一张图来算，遍历文件夹中所有图像  
    for img in images:
        read_img = os.path.join(img_dir,img)
        image = Image.open(read_img)
        
        #图像预处理，将图像缩放到固定分辨率，并进行标准化
        image = image.resize((input_H, input_W))
        image = np.float32(image) / 255
        input_tensor = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])(image)
        
        #添加batch维度
        input_tensor = input_tensor.unsqueeze(0)
        
        if torch.cuda.is_available():
            model = model.cuda()
            input_tensor = input_tensor.cuda()
            
        #输入张量需要计算梯度
        input_tensor.requires_grad = True
        fmap_block = list()
        input_block = list()
        
        #对指定层获取特征图
        layer.register_forward_hook(farward_hook)
        
        #进行一次正向传播
        output = model(input_tensor)
        
        #特征图的channel维度算均值且去掉batch维度，得到二维张量
        feature_map = fmap_block[0].mean(dim=1,keepdim=False).squeeze()
        # feature_map = fmap_block.mean(dim=1,keepdim=False).squeeze()
        
        #对二维张量中心点（标量）进行backward
        feature_map[(feature_map.shape[0]//2-1)][(feature_map.shape[1]//2-1)].backward(retain_graph=True)

        #对输入层的梯度求绝对值
        grad = torch.abs(input_tensor.grad)
        
        #梯度的channel维度算均值且去掉batch维度，得到二维张量，张量大小为输入图像大小
        grad = grad.mean(dim=1,keepdim=False).squeeze()
        
        #累加所有图像的梯度，由于后面要进行归一化，这里可以不算均值
        heatmap = heatmap + grad.cpu().numpy()
        
        
    cam = heatmap

    #对累加的梯度进行归一化
    cam = cam / cam.max()

    #可视化，蓝色值小，红色值大
    cam = cv2.applyColorMap(np.uint8(cam*255), cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    Image.fromarray(cam)
    plt.imshow(cam)
    plt.show()




    print("end") 


if __name__ == "__main__":

    # model = SCESwinT_SCC_new(dim=128, ngram=2, window_size=8)

    # path = 

    # draw_ERF(model, path, pre_checkpoint, base_check_pos, delta_x, delta_y, min_v, max_v, save_path)


    print("end")











