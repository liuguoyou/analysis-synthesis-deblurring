import os
from pathlib import Path
from glob import glob

import gdown
import matplotlib.pyplot as plt
import numpy as np
import cv2

def get_images_in_dir(dir_path):
    paths = []
    for ext in ('png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'):
        paths.extend(glob(os.path.join(dir_path, '*.'+ext)))
    return paths

def load_image(image_path:str) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        if os.path.exists(image_path):
            raise FileExistsError(f'Failed to load the image because "{image_path}" does not exists')
        else:
            raise ValueError(f'Failed to load the image from "{image_path}"')

    if image.ndim == 2:
        image = np.dstack([image]*3)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def save_image(image_path:str, image:np.ndarray):
    assert image.ndim == 3
    path = Path(image_path)
    os.makedirs(path.parent, exist_ok=True)

    image_path = str(image_path)
    cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def download_analysis_weights(weights_path, force=False):
    if not os.path.exists(weights_path) or force:
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        print('Downloading analysis network weights')
        gdown.download('https://drive.google.com/uc?id=1Nf2bqgZIXzayo5okfsBj5PaaXUPhC9Pu', weights_path, quiet=False)

def download_synthesis_weights(weights_path, force=False):
    if not os.path.exists(weights_path) or force:
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        print('Downloading synthesis network weights')
        gdown.download('https://drive.google.com/uc?id=1YXY1420znxXAOUTPQi4xVWreLGpqJ6C2', weights_path, quiet=False)


def plot_results(image_path, blurry_image, deblurred_image):
    '''Plots the blurry and deblurred images side by side'''
    h, w = blurry_image.shape[:2]
    plt.figure(figsize=(2 * w/100, h/100), dpi=100)
    plt.suptitle(image_path, wrap=True)

    plt.subplot(121)
    plt.imshow(blurry_image)
    plt.title('Input image')

    plt.subplot(122)
    plt.imshow(deblurred_image)
    plt.title('Deblurred image')
    plt.show()


def pad_to_divisible(images_BHWC: np.ndarray, d: int, pad_mode='symmetric', **pad_kwargs):
    '''
    Pads the spatial dimensions (H and W) to be divisible by d, i.e., after padding w % d == 0 and h % d == 0,
    where h, w are the height and width of the padded results respectively

    :return: a tuple padded_image, padding
    '''
    h, w = images_BHWC.shape[1:3]
    if h % d != 0:
        r = d - h % d
        pad_y = (r // 2, int(np.ceil(r / 2)))
    else:
        pad_y = (0, 0)

    if w % d != 0:
        r = d - w % d
        pad_x = (r // 2, int(np.ceil(r / 2)))
    else:
        pad_x = (0, 0)

    padding = ((0, 0), pad_y, pad_x, (0, 0))

    return np.pad(images_BHWC, padding, pad_mode, **pad_kwargs), (pad_y, pad_x)

def remove_padding(images_BHWC, padding):
    '''Removes the padding added by pad_to_divisible'''
    if not np.any(padding):
        return images_BHWC
    pad_y, pad_x = padding
    h, w = images_BHWC.shape[1:3]
    return images_BHWC[:, pad_y[0]:h - pad_y[1], pad_x[0]:w - pad_x[1]]