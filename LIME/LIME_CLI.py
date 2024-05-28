import numpy as np
from scipy import fft
from skimage import io, exposure, img_as_ubyte, img_as_float
from tqdm import trange
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import cv2
import time
def firstOrderDerivative(n, k=1):
    return np.eye(n) * (-1) + np.eye(n, k=k)


def toeplitizMatrix(n, row):
    vecDD = np.zeros(n)
    vecDD[0] = 4
    vecDD[1] = -1
    vecDD[row] = -1
    vecDD[-1] = -1
    vecDD[-row] = -1
    return vecDD


def vectorize(matrix):
    return matrix.T.ravel()


def reshape(vector, row, col):
    return vector.reshape((row, col), order='F')


class LIME:
    def __init__(self, iterations=10, alpha=2, rho=2, gamma=0.7, strategy=2, *args, **kwargs):
        self.iterations = iterations
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
        self.strategy = strategy

    def load(self, imgPath):
        self.L = img_as_float(io.imread(imgPath))
        self.row = self.L.shape[0]
        self.col = self.L.shape[1]

        self.T_hat = np.max(self.L, axis=2)
        self.dv = firstOrderDerivative(self.row)
        self.dh = firstOrderDerivative(self.col, -1)
        self.vecDD = toeplitizMatrix(self.row * self.col, self.row)
        self.W = self.weightingStrategy()

    def weightingStrategy(self):
        if self.strategy == 2:
            dTv = self.dv @ self.T_hat
            dTh = self.T_hat @ self.dh
            Wv = 1 / (np.abs(dTv) + 1)
            Wh = 1 / (np.abs(dTh) + 1)
            return np.vstack([Wv, Wh])
        else:
            return np.ones((self.row * 2, self.col))

    def __T_subproblem(self, G, Z, u):
        X = G - Z / u
        Xv = X[:self.row, :]
        Xh = X[self.row:, :]
        temp = self.dv @ Xv + Xh @ self.dh
        numerator = fft.fft(vectorize(2 * self.T_hat + u * temp))
        denominator = fft.fft(self.vecDD * u) + 2
        T = fft.ifft(numerator / denominator)
        T = np.real(reshape(T, self.row, self.col))
        return exposure.rescale_intensity(T, (0, 1), (0.001, 1))

    def __G_subproblem(self, T, Z, u, W):
        dT = self.__derivative(T)
        epsilon = self.alpha * W / u
        X = dT + Z / u
        return np.sign(X) * np.maximum(np.abs(X) - epsilon, 0)

    def __Z_subproblem(self, T, G, Z, u):
        dT = self.__derivative(T)
        return Z + u * (dT - G)

    def __u_subproblem(self, u):
        return u * self.rho

    def __derivative(self, matrix):
        v = self.dv @ matrix
        h = matrix @ self.dh
        return np.vstack([v, h])

    def illumMap(self):
        T = np.zeros((self.row, self.col))
        G = np.zeros((self.row * 2, self.col))
        Z = np.zeros((self.row * 2, self.col))
        u = 1

        for _ in trange(0, self.iterations):
            T = self.__T_subproblem(G, Z, u)
            G = self.__G_subproblem(T, Z, u, self.W)
            Z = self.__Z_subproblem(T, G, Z, u)
            u = self.__u_subproblem(u)

        return T ** self.gamma

    def enhance(self):
        self.T = self.illumMap()
        self.R = self.L / np.repeat(self.T[:, :, np.newaxis], 3, axis=2)
        self.R = exposure.rescale_intensity(self.R, (0, 1))
        self.R = img_as_ubyte(self.R)
        return self.R


def main(options):
    lime = LIME(**options.__dict__)
    lime.load(options.filePath)
    lime.enhance()
    filename = os.path.split(options.filePath)[-1]
    if options.output:
        savePath = f"{options.output}enhanced_{filename}"
        plt.imsave(savePath, lime.R)
    if options.map:
        savePath = f"{options.output}map_{filename}"
        plt.imsave(savePath, lime.T, cmap='gray')


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-f", "--filePath", default="./data/1.bmp", type=str, help="image path to enhance")
#     parser.add_argument("-m", "--map", action="store_true", help="save illumination map")
#     parser.add_argument("-o", "--output", default="./", type=str, help="output folder")

#     parser.add_argument("-i", "--iterations", default=10, type=int, help="iteration number")
#     parser.add_argument("-a", "--alpha", default=2, type=int, help="parameter of alpha")
#     parser.add_argument("-r", "--rho", default=2, type=int, help="parameter of rho")
#     parser.add_argument("-g", "--gamma", default=0.7, type=int, help="parameter of gamma")
#     parser.add_argument("-s", "--strategy", default=2, type=int, choices=[1, 2], help="weighting strategy")
#     options = parser.parse_args()
#     main(options)

# 定义增强整个数据集的函数
def enhance_dataset(image_folder, gt_folder, output_folder, lime_params):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)
    ssim_total = 0
    psnr_total = 0
    
    for image_file in tqdm(image_files, desc='Enhancing images'):
        image_path = os.path.join(image_folder, image_file)
        gt_path = os.path.join(gt_folder, image_file)  # 确保gt图像路径正确
        start_time = time.time()
        lime = LIME(**lime_params)
        lime.load(image_path)
        enhanced_image = lime.enhance()
        end_time = time.time() 
        elapsed_time = end_time - start_time
        print(f"Image: {image_file}, Processing Time: {elapsed_time:.2f} seconds")
        output_path = os.path.join(output_folder, f"LIME_{image_file}")
        plt.imsave(output_path, img_as_ubyte(enhanced_image))

    #     gt_image = img_as_float(io.imread(gt_path))
    #     if enhanced_image.shape != gt_image.shape:
    #         # 使用cv2.resize来调整增强图像的大小以匹配gt_image
    #         enhanced_image = cv2.resize(enhanced_image, (gt_image.shape[1], gt_image.shape[0]), interpolation=cv2.INTER_AREA)

    #     # 计算SSIM和PSNR
    #     ssim_value = ssim(gt_image, enhanced_image, data_range=1, multichannel=True, channel_axis=-1)
    #     psnr_value = psnr(gt_image, enhanced_image, data_range=1)

    #     print(f"Image: {image_file}, SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.4f}")

    #     ssim_total += ssim_value
    #     psnr_total += psnr_value

    # avg_ssim = ssim_total / total_images
    # avg_psnr = psnr_total / total_images
    # print(f"Average SSIM: {avg_ssim:.4f}, Average PSNR: {avg_psnr:.4f}")

# 设置LIME参数
lime_params = {
    "iterations": 10,
    "alpha": 2,
    "rho": 2,
    "gamma": 0.7,
    "strategy": 2,
    "filePath": "",  # 将为每张图像设置路径
    "output": "",  # 将为输出设置路径
    "map": False  # 是否保存照明图
}

# 设置图像输入和输出路径
image_folder = '/home/suxin/mambaeec/data2/test/input/'
gt_folder = '/home/suxin/mambaeec/data2/test/gt/'
output_folder = '/home/suxin/LIME/output_2/'

# 增强整个数据集
enhance_dataset(image_folder, gt_folder, output_folder, lime_params)