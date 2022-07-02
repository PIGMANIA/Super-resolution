import random
import numpy as np
import cv2
import math
import torch

from scipy import special
from scipy import ndimage
from scipy.stats import multivariate_normal
from scipy.linalg import orth

from utils import uint2single, single2uint


class Degradation:
    def __init__(self, cfg):
        self.sf = cfg.scale
        self.shuffle_prob = cfg.deg.shuffle_prob
        self.num_deg = 9

        # Sharpen
        self.sharpen = cfg.sharpen.use
        self.clahe = cfg.clahe.use
        self.sharpen_weight = cfg.sharpen.weight
        self.sharpen_radius = cfg.sharpen.radius
        self.sharpen_threshold = cfg.sharpen.threshold

        """ degradation """
        self.deg = cfg.deg.use

        # Sinc
        self.sinc_prob = cfg.deg.sinc_prob
        self.sinc_prob2 = cfg.deg.sinc_prob2

        # Blur
        self.kernel_list = cfg.deg.kernel_list
        self.kernel_prob = cfg.deg.kernel_prob
        self.blur_sigma = cfg.deg.blur_sigma
        self.betag_range = cfg.deg.betag_range
        self.betap_range = cfg.deg.betap_range
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]

        self.kernel_list2 = cfg.deg.kernel_list2
        self.kernel_prob2 = cfg.deg.kernel_prob2
        self.blur_sigma2 = cfg.deg.blur_sigma2
        self.betag_range2 = cfg.deg.betag_range2
        self.betap_range2 = cfg.deg.betap_range2

        # Resize
        self.resize_prob = cfg.deg.resize_prob
        self.resize_range = cfg.deg.resize_range
        self.resize_prob2 = cfg.deg.resize_prob2
        self.resize_range2 = cfg.deg.resize_range2

        self.updown_type = cfg.deg.updown_type
        self.mode_list = cfg.deg.mode_list

        # Noise
        self.noise_level1 = cfg.deg.noise_level1  # 2
        self.noise_level2 = cfg.deg.noise_level2  # 25

        # Sinc
        self.pulse_tensor = torch.zeros(
            21, 21
        ).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def degradation_pipeline(self, hr):
        hr = uint2single(np.array(hr))
        lr = hr.copy()

        if self.clahe:
            hr = self.add_clahe(hr)

        if self.sharpen:
            hr = self.add_sharpen(hr)

        if random.random() < self.shuffle_prob:
            shuffle_order = random.sample(range(self.num_deg), self.num_deg)
        else:
            shuffle_order = list(range(self.num_deg))
            # local shuffle for noise, JPEG is always the last one
            shuffle_order[2:4] = random.sample(shuffle_order[2:4], len(range(2, 4)))
            shuffle_order[7 : self.num_deg] = random.sample(
                shuffle_order[7 : self.num_deg], len(range(7, self.num_deg))
            )

        if self.deg:
            for i in shuffle_order:
                ### first phase of degradation
                if i == 0:
                    lr = self.generate_kernel1(lr)
                elif i == 1:
                    lr = self.random_resizing(lr)
                elif i == 2:
                    lr = self.add_Poisson_noise(lr)
                elif i == 3:
                    lr = self.add_Gaussian_noise(lr)
                elif i == 4:
                    lr = self.add_speckle_noise(lr)

                ### second phase of degradation
                elif i == 5:
                    lr = self.generate_kernel2(lr)
                elif i == 6:
                    lr = self.random_resizing2(lr)
                elif i == 7:
                    lr = self.add_Poisson_noise(lr)
                elif i == 8:
                    lr = self.add_Gaussian_noise(lr)
                elif i == 9:
                    lr = self.add_speckle_noise(lr)

            if np.random.uniform() < 0.5:
                lr = self.generate_sinc(lr)
                lr = self.add_JPEG_noise(lr)
            else:
                lr = self.add_JPEG_noise(lr)
                lr = self.generate_sinc(lr)

        lr = cv2.resize(
            lr,
            (int(1 / self.sf * hr.shape[1]), int(1 / self.sf * hr.shape[0])),
            interpolation=random.choice([1, 2, 3]),
        )

        lr = single2uint(lr)
        hr = single2uint(hr)
        return lr, hr

    def random_resizing(self, image):
        h, w, c = image.shape

        updown_type = random.choices(self.updown_type, self.resize_prob)
        mode = random.choice(self.mode_list)

        if updown_type == "up":
            scale = np.random.uniform(1, self.resize_range(1))
        elif updown_type == "down":
            scale = np.random.uniform(self.resize_range[0], 1)
        else:
            scale = 1

        if mode == "area":
            flags = cv2.INTER_AREA
        elif mode == "bilinear":
            flags = cv2.INTER_LINEAR
        elif mode == "bicubic":
            flags = cv2.INTER_CUBIC

        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=flags)
        return image

    def random_resizing2(self, image):
        h, w, c = image.shape
        updown_type = random.choices(self.updown_type, self.resize_prob2)
        mode = random.choice(self.mode_list)

        if updown_type == "up":
            scale = np.random.uniform(1, self.resize_range2[1])
        elif updown_type == "down":
            scale = np.random.uniform(self.resize_range2[0], 1)
        else:
            scale = 1

        if mode == "area":
            flags = cv2.INTER_AREA
        elif mode == "bilinear":
            flags = cv2.INTER_LINEAR
        elif mode == "bicubic":
            flags = cv2.INTER_CUBIC

        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=flags)
        return image

    def random_mixed_kernels(
        self,
        kernel_list,
        kernel_prob,
        kernel_size=21,
        sigma_x_range=[0.6, 5],
        sigma_y_range=[0.6, 5],
        rotation_range=[-math.pi, math.pi],
        betag_range=[0.5, 8],
        betap_range=[0.5, 8],
        noise_range=None,
    ):
        """Randomly generate mixed kernels.
        Args:
            kernel_list (tuple): a list name of kenrel types,
                support ['iso', 'aniso', 'skew', 'generalized', 'plateau_iso',
                'plateau_aniso']
            kernel_prob (tuple): corresponding kernel probability for each
                kernel type
            kernel_size (int):
            sigma_x_range (tuple): [0.6, 5]
            sigma_y_range (tuple): [0.6, 5]
            rotation range (tuple): [-math.pi, math.pi]
            beta_range (tuple): [0.5, 8]
            noise_range(tuple, optional): multiplicative kernel noise,
                [0.75, 1.25]. Default: None
        Returns:
            kernel (ndarray):
        """
        kernel_type = random.choices(kernel_list, kernel_prob)[0]
        if kernel_type == "iso":
            kernel = self.random_bivariate_Gaussian(
                kernel_size,
                sigma_x_range,
                sigma_y_range,
                rotation_range,
                noise_range=noise_range,
                isotropic=True,
            )
        elif kernel_type == "aniso":
            kernel = self.random_bivariate_Gaussian(
                kernel_size,
                sigma_x_range,
                sigma_y_range,
                rotation_range,
                noise_range=noise_range,
                isotropic=False,
            )
        elif kernel_type == "generalized_iso":
            kernel = self.random_bivariate_generalized_Gaussian(
                kernel_size,
                sigma_x_range,
                sigma_y_range,
                rotation_range,
                betag_range,
                noise_range=noise_range,
                isotropic=True,
            )
        elif kernel_type == "generalized_aniso":
            kernel = self.random_bivariate_generalized_Gaussian(
                kernel_size,
                sigma_x_range,
                sigma_y_range,
                rotation_range,
                betag_range,
                noise_range=noise_range,
                isotropic=False,
            )
        elif kernel_type == "plateau_iso":
            kernel = self.random_bivariate_plateau(
                kernel_size,
                sigma_x_range,
                sigma_y_range,
                rotation_range,
                betap_range,
                noise_range=None,
                isotropic=True,
            )
        elif kernel_type == "plateau_aniso":
            kernel = self.random_bivariate_plateau(
                kernel_size,
                sigma_x_range,
                sigma_y_range,
                rotation_range,
                betap_range,
                noise_range=None,
                isotropic=False,
            )
        return kernel

    def random_bivariate_Gaussian(
        self,
        kernel_size,
        sigma_x_range,
        sigma_y_range,
        rotation_range,
        noise_range=None,
        isotropic=True,
    ):
        """Randomly generate bivariate isotropic or anisotropic Gaussian kernels.
        In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.
        Args:
            kernel_size (int):
            sigma_x_range (tuple): [0.6, 5]
            sigma_y_range (tuple): [0.6, 5]
            rotation range (tuple): [-math.pi, math.pi]
            noise_range(tuple, optional): multiplicative kernel noise,
                [0.75, 1.25]. Default: None
        Returns:
            kernel (ndarray):
        """
        assert kernel_size % 2 == 1, "Kernel size must be an odd number."
        assert sigma_x_range[0] < sigma_x_range[1], "Wrong sigma_x_range."
        sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
        if isotropic is False:
            assert sigma_y_range[0] < sigma_y_range[1], "Wrong sigma_y_range."
            assert rotation_range[0] < rotation_range[1], "Wrong rotation_range."
            sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
            rotation = np.random.uniform(rotation_range[0], rotation_range[1])
        else:
            sigma_y = sigma_x
            rotation = 0

        kernel = self.bivariate_Gaussian(
            kernel_size, sigma_x, sigma_y, rotation, isotropic=isotropic
        )

        # add multiplicative noise
        if noise_range is not None:
            assert noise_range[0] < noise_range[1], "Wrong noise range."
            noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
            kernel = kernel * noise
        kernel = kernel / np.sum(kernel)
        return kernel

    def random_bivariate_generalized_Gaussian(
        self,
        kernel_size,
        sigma_x_range,
        sigma_y_range,
        rotation_range,
        beta_range,
        noise_range=None,
        isotropic=True,
    ):
        """Randomly generate bivariate generalized Gaussian kernels.
        In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.
        Args:
            kernel_size (int):
            sigma_x_range (tuple): [0.6, 5]
            sigma_y_range (tuple): [0.6, 5]
            rotation range (tuple): [-math.pi, math.pi]
            beta_range (tuple): [0.5, 8]
            noise_range(tuple, optional): multiplicative kernel noise,
                [0.75, 1.25]. Default: None
        Returns:
            kernel (ndarray):
        """
        assert kernel_size % 2 == 1, "Kernel size must be an odd number."
        assert sigma_x_range[0] < sigma_x_range[1], "Wrong sigma_x_range."
        sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
        if isotropic is False:
            assert sigma_y_range[0] < sigma_y_range[1], "Wrong sigma_y_range."
            assert rotation_range[0] < rotation_range[1], "Wrong rotation_range."
            sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
            rotation = np.random.uniform(rotation_range[0], rotation_range[1])
        else:
            sigma_y = sigma_x
            rotation = 0

        # assume beta_range[0] < 1 < beta_range[1]
        if np.random.uniform() < 0.5:
            beta = np.random.uniform(beta_range[0], 1)
        else:
            beta = np.random.uniform(1, beta_range[1])

        kernel = self.bivariate_generalized_Gaussian(
            kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic
        )

        # add multiplicative noise
        if noise_range is not None:
            assert noise_range[0] < noise_range[1], "Wrong noise range."
            noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
            kernel = kernel * noise
        kernel = kernel / np.sum(kernel)
        return kernel

    def random_bivariate_plateau(
        self,
        kernel_size,
        sigma_x_range,
        sigma_y_range,
        rotation_range,
        beta_range,
        noise_range=None,
        isotropic=True,
    ):
        """Randomly generate bivariate plateau kernels.
        In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.
        Args:
            kernel_size (int):
            sigma_x_range (tuple): [0.6, 5]
            sigma_y_range (tuple): [0.6, 5]
            rotation range (tuple): [-math.pi/2, math.pi/2]
            beta_range (tuple): [1, 4]
            noise_range(tuple, optional): multiplicative kernel noise,
                [0.75, 1.25]. Default: None
        Returns:
            kernel (ndarray):
        """
        assert kernel_size % 2 == 1, "Kernel size must be an odd number."
        assert sigma_x_range[0] < sigma_x_range[1], "Wrong sigma_x_range."
        sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
        if isotropic is False:
            assert sigma_y_range[0] < sigma_y_range[1], "Wrong sigma_y_range."
            assert rotation_range[0] < rotation_range[1], "Wrong rotation_range."
            sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
            rotation = np.random.uniform(rotation_range[0], rotation_range[1])
        else:
            sigma_y = sigma_x
            rotation = 0

        # TODO: this may be not proper
        if np.random.uniform() < 0.5:
            beta = np.random.uniform(beta_range[0], 1)
        else:
            beta = np.random.uniform(1, beta_range[1])

        kernel = self.bivariate_plateau(
            kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic
        )
        # add multiplicative noise
        if noise_range is not None:
            assert noise_range[0] < noise_range[1], "Wrong noise range."
            noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
            kernel = kernel * noise
        kernel = kernel / np.sum(kernel)

        return kernel

    def bivariate_Gaussian(
        self, kernel_size, sig_x, sig_y, theta, grid=None, isotropic=True
    ):
        """Generate a bivariate isotropic or anisotropic Gaussian kernel.
        In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.
        Args:
            kernel_size (int):
            sig_x (float):
            sig_y (float):
            theta (float): Radian measurement.
            grid (ndarray, optional): generated by :func:`mesh_grid`,
                with the shape (K, K, 2), K is the kernel size. Default: None
            isotropic (bool):
        Returns:
            kernel (ndarray): normalized kernel.
        """
        if grid is None:
            grid, _, _ = self.mesh_grid(kernel_size)
        if isotropic:
            sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
        else:
            sigma_matrix = self.sigma_matrix2(sig_x, sig_y, theta)
        kernel = self.pdf2(sigma_matrix, grid)
        kernel = kernel / np.sum(kernel)
        return kernel

    def bivariate_generalized_Gaussian(
        self, kernel_size, sig_x, sig_y, theta, beta, grid=None, isotropic=True
    ):
        """Generate a bivariate generalized Gaussian kernel.
            Described in `Parameter Estimation For Multivariate Generalized
            Gaussian Distributions`_
            by Pascal et. al (2013).
        In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.
        Args:
            kernel_size (int):
            sig_x (float):
            sig_y (float):
            theta (float): Radian measurement.
            beta (float): shape parameter, beta = 1 is the normal distribution.
            grid (ndarray, optional): generated by :func:`mesh_grid`,
                with the shape (K, K, 2), K is the kernel size. Default: None
        Returns:
            kernel (ndarray): normalized kernel.
        .. _Parameter Estimation For Multivariate Generalized Gaussian
        Distributions: https://arxiv.org/abs/1302.6498
        """
        if grid is None:
            grid, _, _ = self.mesh_grid(kernel_size)
        if isotropic:
            sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
        else:
            sigma_matrix = self.sigma_matrix2(sig_x, sig_y, theta)
        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.exp(
            -0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta)
        )
        kernel = kernel / np.sum(kernel)
        return kernel

    def bivariate_plateau(
        self, kernel_size, sig_x, sig_y, theta, beta, grid=None, isotropic=True
    ):
        """Generate a plateau-like anisotropic kernel.
        1 / (1+x^(beta))
        Ref: https://stats.stackexchange.com/questions/203629/is-there-a-plateau-shaped-distribution
        In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.
        Args:
            kernel_size (int):
            sig_x (float):
            sig_y (float):
            theta (float): Radian measurement.
            beta (float): shape parameter, beta = 1 is the normal distribution.
            grid (ndarray, optional): generated by :func:`mesh_grid`,
                with the shape (K, K, 2), K is the kernel size. Default: None
        Returns:
            kernel (ndarray): normalized kernel.
        """
        if grid is None:
            grid, _, _ = self.mesh_grid(kernel_size)
        if isotropic:
            sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
        else:
            sigma_matrix = self.sigma_matrix2(sig_x, sig_y, theta)
        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.reciprocal(
            np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta) + 1
        )
        kernel = kernel / np.sum(kernel)
        return kernel

    def circular_lowpass_kernel(self, cutoff, kernel_size, pad_to=0):
        """2D sinc filter, ref: https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter

        Args:
            cutoff (float): cutoff frequency in radians (pi is max)
            kernel_size (int): horizontal and vertical size, must be odd.
            pad_to (int): pad kernel size to desired size, must be odd or zero.
        """
        assert kernel_size % 2 == 1, "Kernel size must be an odd number."
        kernel = np.fromfunction(
            lambda x, y: cutoff
            * special.j1(
                cutoff
                * np.sqrt(
                    (x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2
                )
            )
            / (
                2
                * np.pi
                * np.sqrt(
                    (x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2
                )
            ),
            [kernel_size, kernel_size],
        )
        kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff**2 / (
            4 * np.pi
        )
        kernel = kernel / np.sum(kernel)
        if pad_to > kernel_size:
            pad_size = (pad_to - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        return kernel

    def mesh_grid(self, kernel_size):
        """Generate the mesh grid, centering at zero.
        Args:
            kernel_size (int):
        Returns:
            xy (ndarray): with the shape (kernel_size, kernel_size, 2)
            xx (ndarray): with the shape (kernel_size, kernel_size)
            yy (ndarray): with the shape (kernel_size, kernel_size)
        """
        ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = np.meshgrid(ax, ax)
        xy = np.hstack(
            (
                xx.reshape((kernel_size * kernel_size, 1)),
                yy.reshape(kernel_size * kernel_size, 1),
            )
        ).reshape(kernel_size, kernel_size, 2)
        return xy, xx, yy

    def pdf2(self, sigma_matrix, grid):
        """Calculate PDF of the bivariate Gaussian distribution.
        Args:
            sigma_matrix (ndarray): with the shape (2, 2)
            grid (ndarray): generated by :func:`mesh_grid`,
                with the shape (K, K, 2), K is the kernel size.
        Returns:
            kernel (ndarrray): un-normalized kernel.
        """
        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
        return kernel

    def sigma_matrix2(self, sig_x, sig_y, theta):
        """Calculate the rotated sigma matrix (two dimensional matrix).
        Args:
            sig_x (float):
            sig_y (float):
            theta (float): Radian measurement.
        Returns:
            ndarray: Rotated sigma matrix.
        """
        D = np.array([[sig_x**2, 0], [0, sig_y**2]])
        U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return np.dot(U, np.dot(D, U.T))

    def generate_kernel1(self, image):
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = self.circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = self.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None,
            )

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        image = ndimage.filters.convolve(
            image, np.expand_dims(kernel, axis=2), mode="reflect"
        )
        return image.clip(min=0, max=255)

    def generate_kernel2(self, image):
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = self.circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = self.random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2,
                [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None,
            )

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        image = ndimage.filters.convolve(
            image, np.expand_dims(kernel, axis=2), mode="reflect"
        )
        return image.clip(min=0, max=255)

    def add_clahe(self, img):
        img = single2uint(img)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        img = uint2single(img)
        return img

    def add_sharpen(self, img):
        """USM sharpening. borrowed from real-ESRGAN
        Input image: I; Blurry image: B.
        1. K = I + weight * (I - B)
        2. Mask = 1 if abs(I - B) > threshold, else: 0
        3. Blur mask:
        4. Out = Mask * K + (1 - Mask) * I
        Args:
            img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
            weight (float): Sharp weight. Default: 1.
            radius (float): Kernel size of Gaussian blur. Default: 50.
            threshold (int):
        """
        # if self.sharpen_radius % 2 == 0:
        #     self.sharpen_radius += 1
        # blur = cv2.GaussianBlur(img, (self.sharpen_radius, self.sharpen_radius), 0)
        # residual = img - blur
        # mask = np.abs(residual) * 255 > self.sharpen_threshold
        # mask = mask.astype("float32")
        # soft_mask = cv2.GaussianBlur(
        #     mask, (self.sharpen_radius, self.sharpen_radius), 0
        # )

        # K = img + self.sharpen_weight * residual
        # K = np.clip(K, 0, 1)
        # return soft_mask * K + (1 - soft_mask) * img
        img = single2uint(img)
        kernel_3 = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel_3)
        img = uint2single(img)
        return img

    def add_JPEG_noise(self, img):
        quality_factor = random.randint(30, 95)
        img = cv2.cvtColor(single2uint(img), cv2.COLOR_RGB2BGR)
        result, encimg = cv2.imencode(
            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor]
        )
        img = cv2.imdecode(encimg, 1)
        img = cv2.cvtColor(uint2single(img), cv2.COLOR_BGR2RGB)
        return img

    def add_Gaussian_noise(self, img):
        noise_level = random.randint(self.noise_level1, self.noise_level2)
        rnum = np.random.rand()
        if rnum > 0.6:  # add color Gaussian noise
            img += np.random.normal(0, noise_level / 255.0, img.shape).astype(
                np.float32
            )
        elif rnum < 0.4:  # add grayscale Gaussian noise
            img += np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(
                np.float32
            )
        else:  # add  noise
            L = self.noise_level2 / 255.0
            D = np.diag(np.random.rand(3))
            U = orth(np.random.rand(3, 3))
            conv = np.dot(np.dot(np.transpose(U), D), U)
            img += np.random.multivariate_normal(
                [0, 0, 0], np.abs(L**2 * conv), img.shape[:2]
            ).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)
        return img

    def add_speckle_noise(self, img):
        noise_level = random.randint(self.noise_level1, self.noise_level2)
        img = np.clip(img, 0.0, 1.0)
        rnum = random.random()
        if rnum > 0.6:
            img += img * np.random.normal(0, noise_level / 255.0, img.shape).astype(
                np.float32
            )
        elif rnum < 0.4:
            img += img * np.random.normal(
                0, noise_level / 255.0, (*img.shape[:2], 1)
            ).astype(np.float32)
        else:
            L = self.noise_level2 / 255.0
            D = np.diag(np.random.rand(3))
            U = orth(np.random.rand(3, 3))
            conv = np.dot(np.dot(np.transpose(U), D), U)
            img += img * np.random.multivariate_normal(
                [0, 0, 0], np.abs(L**2 * conv), img.shape[:2]
            ).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)
        return img

    def add_Poisson_noise(self, img):
        vals = 10 ** (2 * random.random() + 2.0)  # [2, 4]
        if random.random() < 0.5:
            img = np.random.poisson(img * vals).astype(np.float32) / vals
        else:
            img_gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
            img_gray = np.clip((img_gray * 255.0).round(), 0, 255) / 255.0
            noise_gray = (
                np.random.poisson(img_gray * vals).astype(np.float32) / vals - img_gray
            )
            img += noise_gray[:, :, np.newaxis]
        img = np.clip(img, 0.0, 1.0)
        return img

    def generate_sinc(self, image):
        # if np.random.uniform() < self.final_sinc_prob:
        #     kernel_size = random.choice(self.kernel_range)
        #     omega_c = np.random.uniform(np.pi / 3, np.pi)
        #     sinc_kernel = self.circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        #     sinc_kernel = torch.FloatTensor(sinc_kernel)
        # else:
        sinc_kernel = self.pulse_tensor

        image = ndimage.filters.convolve(
            image, np.expand_dims(sinc_kernel, axis=2), mode="reflect"
        )
        return image.clip(min=0, max=255)
