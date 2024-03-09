import numpy as np
from matplotlib import pyplot as plt

from skimage.metrics import structural_similarity as ssim


def mse(imageA, imageB):
    """ returns the Mean Squared Error between two images, 
    however difficult to standardize due to high scores"""

    # formula is calculated by the sum of the squared differences
    error = np.sum((imageA.astype('float') - imageB.astype('float')) ** 2)
    error /= float(imageA.shape[0] * imageA.shape[1])

    # smaller the error value, the similar the images
    return error


def compare_images(imageA, imageB):
    """ compares two images, returns Structural Similarity Index (SSIM)
    which compares pixel similarity within location, density"""

    score = ssim(imageA, imageB)
    fig = plt.figure(title)
    plt.suptitle('SSIM: %.2f' % score)

    # shows first spectrogram
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis('off')

    # shows second spectrogram
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis('off')