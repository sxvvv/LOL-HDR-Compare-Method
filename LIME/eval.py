import os
import numpy as np
from skimage import io, exposure, img_as_ubyte, img_as_float
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

import warnings
warnings.filterwarnings('ignore')

def calculate_psnr_ssim(clear_img_path, enhanced_img_path, clear_img_names):
    """
    计算清晰图像和增强图像之间的PSNR和SSIM值。
    """
    SSIM_list = []
    PSNR_list = []

    # 确保clear_img_names中的文件名不包含前缀LIME_
    clear_img_names = [name.replace('LIME_', '') for name in clear_img_names]

    for i in range(len(clear_img_names)):
        # 读取清晰图像和增强后的图像
        clear_img = io.imread(os.path.join(clear_img_path, clear_img_names[i]))
        enhanced_img = io.imread(os.path.join(enhanced_img_path, f'LIME_{clear_img_names[i]}'))

        # 计算PSNR
        PSNR = psnr(clear_img, enhanced_img)
        print(f'{i + 1} PSNR: {PSNR}')
        PSNR_list.append(PSNR)

        # 计算SSIM
        SSIM = ssim(clear_img, enhanced_img, channel_axis=2)
        print(f'{i + 1} SSIM: {SSIM}')
        SSIM_list.append(SSIM)

    print(f"average SSIM: {np.mean(SSIM_list)}")
    print(f"average PSNR: {np.mean(PSNR_list)}")

# 测试函数
clear_img_path = '/home/suxin/mambaeec/data2/test/gt'  # 清晰图像文件夹路径
enhanced_img_path = '/home/suxin/LIME/output_2/'  # 增强后的图像文件夹路径
clear_img_names = os.listdir(clear_img_path)  # 清晰图像文件名列表

calculate_psnr_ssim(clear_img_path, enhanced_img_path, clear_img_names)