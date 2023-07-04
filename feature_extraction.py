import numpy as np
from skimage.feature import local_binary_pattern
from skimage import io
from typing import Union

import cv2


def lbp(image: Union[np.ndarray, str], radius: int = 3,  n_points: int = 8, methods: str = 'uniform') -> np.ndarray:
    """
    Calculate Local Binary Pattern of an image.
    :param image: input image
    :param radius: radius of LBP
    :param n_points: number of points
    :return: LBP histogram
    """
    if image is None:
        raise ValueError("image is None")
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    lbp = local_binary_pattern(image, n_points, radius, method=methods)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    # normalize the histogram
    # hist = hist.astype('float')
    # hist /= (hist.sum() + 1e-7)

    return hist.astype('float')

def color_moment(image: Union[np.ndarray, str]) -> np.ndarray:
    """
    Calculate color moment of an image (mean, standard deviation, skewness, kurtosis).
    :param image: input image
    :return: color moment
    """
    if image is None:
        raise ValueError("image is None")
    if isinstance(image, str):
        image = io.imread(image)
    # mean of each channel
    mean = np.mean(image, axis=(0, 1))
    # standard deviation of each channel
    std = np.std(image, axis=(0, 1))
    # skewness of each channel
    skew = np.mean(np.power(image - mean, 3)) / np.power(std, 3)
    # kurtosis of each channel
    kurt = np.mean(np.power(image - mean, 4)) / np.power(std, 4)
    return np.concatenate((mean, std, skew, kurt))

def hu_moment(image: Union[np.ndarray, str]) -> np.ndarray:
    """
    Calculate Hu moment of an image.
    :param image: input image
    :return: Hu moment
    """
    if image is None:
        raise ValueError("image is None")
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    # compute moments
    moments = cv2.HuMoments(cv2.moments(image)).flatten()
    return moments

lbp_arr = hu_moment('data/cleaned_dataset/infected/DSC03984_4.JPG')
print(lbp_arr)