import numpy as np
from astropy.convolution import RickerWavelet2DKernel
from scipy.ndimage.filters import gaussian_filter


def choose_kernel(kernel_type, size=3):
    # print(size)
    if kernel_type == 'gaussian':
        kernel = gaussian_kernel2(size=size, sigma=size / 3)
        # kernel = np.array(matlab_style_gauss2D((110, 11), 3))
        # plt.imshow(kernel)
        # plt.show()
        return kernel
    if kernel_type == 'mexican_hat':
        kernel = ricker_kernel(size=size)
        # plt.imshow(kernel)
        # plt.show()
        return kernel

def gaussian_kernel(size, sigma=1):
    if size % 2 == 0:
        size += 1
    k = np.zeros((size, size))
    k[size // 2, size // 2] = 1
    k = gaussian_filter(k, sigma)
    return k

def gaussian_kernel2(size, sigma=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def __matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def ricker_kernel(size=3, sigma=1.):
    return RickerWavelet2DKernel(width=sigma, x_size=size, y_size=size)
    # return RickerWavelet2DKernel(width=sigma, x_size=51, y_size=11)