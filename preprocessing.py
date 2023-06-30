import os
import sys
from typing import Union

import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image file or image directory")
ap.add_argument("-o", "--output-dir", required=True, help="path to output directory")
ap.add_argument("-e", '--enhance', action='store_true', help="enhance image")
ap.add_argument("-n", '--remove-noise', action='store_true', help="remove noise from image")
ap.add_argument("-r", '--resize', action='store_true', help="resize image")
args = vars(ap.parse_args())


def enhance(image: Union[str, np.ndarray]) -> np.ndarray:
    """
    Enhance image by using adaptive histogram equalization
    :param image: [str, np.ndarray], path to image file or image array in numpy array
    :return: np.ndarray, numpy array of enhance image
    """
    if type(image) is str:
        image = cv2.imread(image)
    # convert from BGR to YCrCb
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    # create clahe object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # equalize the histogram of the Y channel
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    # convert the YCR_CB image back to RGB format
    equalized_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return equalized_image


def remove_noise(image: Union[str, np.ndarray]) -> np.ndarray:
    """
    Remove noise from image using bilateral filter
    :param image: [str, np.ndarray], path to image file or image array in numpy array
    :return: np.ndarray, numpy array of image with removed noise
    """
    if type(image) is str:
        image = cv2.imread(image)
    # apply bilateral filter with d = 15, sigmaColor = sigmaSpace = 75.
    bilateral_filtered_image = cv2.bilateralFilter(image, 15, 75, 75)
    return bilateral_filtered_image


def resize(image: Union[str, np.ndarray], size: int = 256) -> np.ndarray:
    """
    Resize image
    :param image: [str, np.ndarray], path to image file or image array in numpy array
    :param size: int, size of image
    :return: np.ndarray, numpy array of image with removed noise
    """
    if type(image) is str:
        image = cv2.imread(image)
    resized_image = cv2.resize(image, (size, size))
    return resized_image


def preprocess(image: Union[str, np.ndarray]) -> np.ndarray:
    if type(image) is str:
        image = cv2.imread(image)
    if args['enhance']:
        image = enhance(image)
    if args['remove_noise']:
        image = remove_noise(image)
    if args['resize']:
        image = resize(image)
    return image


if __name__ == '__main__':
    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])

    if os.path.isdir(args['image']):
        images_file = os.listdir(args['image'])
        for image_file in images_file:
            image_path = os.path.join(args['image'], image_file)
            image_array = preprocess(image_path)
            sys.stdout.write(f"\rWriting {image_file} to {args['output_dir']}")
            sys.stdout.flush()
            cv2.imwrite(os.path.join(args['output_dir'], image_file), image_array)

    elif os.path.isfile(args['image']):
        image_array = preprocess(args['image'])
        sys.stdout.write(f"\rWriting {os.path.split(args['image'])[-1]} to {args['output_dir']}")
        cv2.imwrite(os.path.join(args['output_dir'], os.path.split(args['image'])[-1]), image_array)

    print("\nDone!")
